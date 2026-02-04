import json
import tempfile
import unittest
from pathlib import Path

import sys

TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import catalog_daemon  # noqa: E402


class _DummyServiceRegistry:
    def load(self) -> None:
        return None

    def refresh_instances(self) -> None:
        return None

    def list_services(self):
        return []

    def list_instances(self):
        return []


class TestCatalogDaemonSotaDistillVisibility(unittest.TestCase):
    def test_snapshot_surfaces_only_success_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".pluribus" / "index").mkdir(parents=True, exist_ok=True)

            # One SOTA item
            (root / ".pluribus" / "index" / "sota.ndjson").write_text(
                json.dumps(
                    {
                        "kind": "sota_item",
                        "id": "item-1",
                        "url": "https://example.com",
                        "title": "Example",
                        "org": "org",
                        "region": "us",
                        "type": "paper",
                        "priority": 3,
                        "cadence_days": 14,
                        "tags": ["sota"],
                        "notes": "",
                        "ts": 1.0,
                        "iso": "2025-01-01T00:00:00Z",
                        "provenance": {},
                    },
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )

            # Distillation artifacts: latest is failed, but an older success exists.
            ok_path = root / ".pluribus" / "index" / "distillations" / "sota" / "item-1" / "ok.md"
            bad_path = root / ".pluribus" / "index" / "distillations" / "sota" / "item-1" / "bad.md"
            ok_path.parent.mkdir(parents=True, exist_ok=True)
            ok_path.write_text("SUCCESS\n", encoding="utf-8")
            bad_path.write_text("FAILED\n## stderr\nhttp error: 429\n", encoding="utf-8")

            artifacts_path = root / ".pluribus" / "index" / "artifacts.ndjson"
            artifacts_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "sota_distillation",
                                "sota_item_id": "item-1",
                                "req_id": "req-ok",
                                "path": str(ok_path),
                                "exit_code": 0,
                                "ts": 2.0,
                                "iso": "2025-01-01T00:00:02Z",
                            },
                            separators=(",", ":"),
                        ),
                        json.dumps(
                            {
                                "type": "sota_distillation",
                                "sota_item_id": "item-1",
                                "req_id": "req-bad",
                                "path": str(bad_path),
                                "exit_code": 1,
                                "ts": 3.0,
                                "iso": "2025-01-01T00:00:03Z",
                            },
                            separators=(",", ":"),
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            emitted: list[tuple[str, dict]] = []
            original_emit = catalog_daemon.emit_bus

            def _capture(bus_dir: Path, topic: str, kind: str, level: str, data: dict) -> None:
                emitted.append((topic, data))

            try:
                catalog_daemon.emit_bus = _capture
                d = catalog_daemon.CatalogDaemon(root=root, bus_dir=root / ".pluribus" / "bus", interval_s=10.0)
                d.svc_reg = _DummyServiceRegistry()
                d.publish_snapshot()
            finally:
                catalog_daemon.emit_bus = original_emit

            sota_events = [d for (topic, d) in emitted if topic == "sota.list"]
            self.assertEqual(len(sota_events), 1)
            items = sota_events[0]["items"]
            self.assertEqual(len(items), 1)
            item = items[0]

            # Status reflects latest attempt (failed), but artifact path/snippet reflect latest success.
            self.assertEqual(item.get("distill_status"), "failed")
            self.assertEqual(item.get("distill_req_id"), "req-bad")
            self.assertEqual(item.get("distill_artifact_path"), str(ok_path))
            self.assertIn("SUCCESS", item.get("distill_snippet") or "")

    def test_failed_response_does_not_emit_artifact_event(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            idx = root / ".pluribus" / "index"
            idx.mkdir(parents=True, exist_ok=True)

            responses_path = idx / "responses.ndjson"
            resp = {
                "id": "resp-1",
                "kind": "strp_response",
                "req_id": "r1",
                "provider": "auto",
                "model": None,
                "exit_code": 1,
                "iso": "2025-01-01T00:00:00Z",
                "output": "",
                "stderr": "http error: 429",
                "ts": 1.0,
            }
            responses_path.write_text(json.dumps(resp, separators=(",", ":")) + "\n", encoding="utf-8")

            emitted: list[tuple[str, dict]] = []
            original_emit = catalog_daemon.emit_bus

            def _capture(bus_dir: Path, topic: str, kind: str, level: str, data: dict) -> None:
                emitted.append((topic, data))

            try:
                catalog_daemon.emit_bus = _capture
                d = catalog_daemon.CatalogDaemon(root=root, bus_dir=root / ".pluribus" / "bus", interval_s=10.0)
                d.svc_reg = _DummyServiceRegistry()
                d._sota_pending["r1"] = "item-1"
                d.check_new_responses()
            finally:
                catalog_daemon.emit_bus = original_emit

            topics = [t for (t, _) in emitted]
            self.assertIn("sota.distill.status", topics)
            self.assertNotIn("sota.distill.artifact", topics)

            status_payloads = [d for (t, d) in emitted if t == "sota.distill.status"]
            self.assertEqual(len(status_payloads), 1)
            status = status_payloads[0]
            self.assertEqual(status.get("status"), "failed")
            self.assertNotIn("path", status)
            self.assertIn("http error: 429", status.get("error") or "")

            # Artifact is still materialized on disk for debugging/provenance.
            artifact_path = root / ".pluribus" / "index" / "distillations" / "sota" / "item-1" / "r1.md"
            self.assertTrue(artifact_path.exists())


if __name__ == "__main__":
    unittest.main()

