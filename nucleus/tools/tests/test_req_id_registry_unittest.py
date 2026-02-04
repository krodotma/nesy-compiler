import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class TestRequestIdRegistry(unittest.TestCase):
    def test_acquire_collision_release(self) -> None:
        from nucleus.tools.req_id_registry import RequestIdRegistry

        with TemporaryDirectory() as td:
            registry_path = Path(td) / "req_id_registry.json"
            reg = RequestIdRegistry(registry_path=registry_path)

            ok, existing = reg.acquire("req-1", actor="a", topic="t")
            self.assertTrue(ok)
            self.assertIsNone(existing)

            ok, existing = reg.acquire("req-1", actor="b", topic="t")
            self.assertFalse(ok)
            self.assertEqual(existing, "a")

            self.assertTrue(reg.release("req-1"))

            ok, existing = reg.acquire("req-1", actor="b", topic="t")
            self.assertTrue(ok)
            self.assertIsNone(existing)

    def test_stale_entries_are_pruned(self) -> None:
        from nucleus.tools.req_id_registry import RequestIdRegistry, STALE_THRESHOLD_S

        with TemporaryDirectory() as td:
            registry_path = Path(td) / "req_id_registry.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "req-stale": {
                            "req_id": "req-stale",
                            "actor": "old",
                            "acquired_at": time.time() - (STALE_THRESHOLD_S + 5),
                            "topic": "x",
                        }
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            reg = RequestIdRegistry(registry_path=registry_path)

            ok, existing = reg.acquire("req-stale", actor="new", topic="y")
            self.assertTrue(ok)
            self.assertIsNone(existing)


if __name__ == "__main__":
    unittest.main()

