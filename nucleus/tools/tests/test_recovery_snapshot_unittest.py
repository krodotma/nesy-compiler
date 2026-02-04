import unittest
from pathlib import Path
from unittest.mock import patch


class TestRecoverySnapshot(unittest.TestCase):
    def test_build_snapshot_summarizes_status(self) -> None:
        from nucleus.tools.recovery_snapshot import build_snapshot

        snapshot = build_snapshot(
            status_lines=["?? new_file.txt", " M changed_file.txt", "A  added_file.txt"],
            log_data={"commits": [{"sha": "abc", "message": "test"}]},
            bus_lines=["{\"topic\":\"test\"}"],
            ledger_lines=["{\"req_id\":\"req-1\"}"],
            actor="tester",
            cwd="/tmp",
            run_id="run-1",
            created_iso="2025-12-19T00:00:00Z",
        )

        self.assertEqual(snapshot["summary"]["untracked_count"], 1)
        self.assertEqual(snapshot["summary"]["modified_count"], 1)
        self.assertEqual(snapshot["summary"]["added_count"], 1)
        self.assertEqual(snapshot["log"]["commits"][0]["sha"], "abc")
        self.assertEqual(snapshot["actor"], "tester")
        self.assertEqual(snapshot["run_id"], "run-1")

    def test_build_snapshot_defaults(self) -> None:
        from nucleus.tools.recovery_snapshot import build_snapshot

        snapshot = build_snapshot(
            status_lines=[],
            log_data=None,
            bus_lines=[],
            ledger_lines=[],
            actor="tester",
            cwd="/tmp",
            run_id=None,
            created_iso="2025-12-19T00:00:00Z",
        )

        self.assertEqual(snapshot["summary"]["untracked_count"], 0)
        self.assertEqual(snapshot["summary"]["modified_count"], 0)
        self.assertEqual(snapshot["summary"]["added_count"], 0)
        self.assertIsNone(snapshot["log"]) 

    def test_emit_bus_event(self) -> None:
        from nucleus.tools import agent_bus
        from nucleus.tools import recovery_snapshot

        dummy_paths = agent_bus.BusPaths(
            active_dir="/tmp",
            events_path="/tmp/events.ndjson",
            primary_dir="/tmp",
            fallback_dir=None,
        )
        sample_snapshot = {
            "summary": {"untracked_count": 0, "modified_count": 0, "added_count": 0},
            "created_iso": "2025-12-19T00:00:00Z",
        }

        with patch("nucleus.tools.recovery_snapshot.collect_snapshot", return_value=sample_snapshot), \
            patch("nucleus.tools.recovery_snapshot.write_snapshot", return_value=Path("/tmp/snap.json")), \
            patch("nucleus.tools.recovery_snapshot.agent_bus.resolve_bus_paths", return_value=dummy_paths), \
            patch("nucleus.tools.recovery_snapshot.agent_bus.emit_event") as emit:
            with patch("sys.argv", ["recovery_snapshot.py", "--repo-dir", "/tmp"]):
                recovery_snapshot.main()

            self.assertTrue(emit.called)
            kwargs = emit.call_args.kwargs
            self.assertEqual(kwargs["topic"], "recovery.snapshot.created")


if __name__ == "__main__":
    unittest.main()
