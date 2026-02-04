import json
import os
import tempfile
import time
import unittest
from pathlib import Path


class TestOmegaHeartbeatPairs(unittest.TestCase):
    def test_pending_pairs_counts(self):
        from nucleus.tools import omega_heartbeat

        with tempfile.TemporaryDirectory() as td:
            bus_dir = Path(td)
            events = bus_dir / "events.ndjson"
            now = time.time()
            # request without response => pending
            events.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "1",
                                "ts": now - 10,
                                "iso": "2025-12-15T00:00:00Z",
                                "topic": "infer_sync.request",
                                "kind": "request",
                                "level": "info",
                                "actor": "t",
                                "data": {"req_id": "r1"},
                            }
                        ),
                        json.dumps(
                            {
                                "id": "2",
                                "ts": now - 5,
                                "iso": "2025-12-15T00:00:05Z",
                                "topic": "infer_sync.response",
                                "kind": "response",
                                "level": "info",
                                "actor": "t",
                                "data": {"req_id": "r2"},
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            spec = {
                "schema_version": 1,
                "pairs": [
                    {
                        "id": "infer_sync",
                        "request_topics": ["infer_sync.request"],
                        "response_topics": ["infer_sync.response"],
                    }
                ],
            }
            res = omega_heartbeat.count_pending_pairs(str(bus_dir), spec)
            self.assertEqual(res["total"], 1)
            self.assertEqual(res["by_pair"]["infer_sync"]["pending"], 1)

