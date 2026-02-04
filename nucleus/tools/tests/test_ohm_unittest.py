import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class TestOmegaHeartMonitor(unittest.TestCase):
    def test_tail_file_lines(self) -> None:
        from nucleus.tools import ohm

        monitor = ohm.OmegaHeartMonitor(bootstrap_tasks=False)
        with TemporaryDirectory() as td:
            path = Path(td) / "ledger.ndjson"
            path.write_text("a\nb\nc\nd\n", encoding="utf-8")
            lines = monitor._tail_file_lines(str(path), max_bytes=1024, max_lines=2)

        self.assertEqual(lines, ["c", "d"])

    def test_task_counts_active(self) -> None:
        from nucleus.tools import ohm

        monitor = ohm.OmegaHeartMonitor(bootstrap_tasks=False)
        monitor._ingest_task_entry({"id": "1", "run_id": "task-1", "status": "in_progress", "actor": "alpha"})
        monitor._ingest_task_entry({"id": "2", "run_id": "task-2", "status": "blocked", "actor": "beta"})
        monitor._ingest_task_entry({"id": "3", "run_id": "task-1", "status": "completed", "actor": "alpha"})

        counts, active = monitor._task_counts()

        self.assertEqual(counts["completed"], 1)
        self.assertEqual(counts["blocked"], 1)
        self.assertEqual(counts["in_progress"], 0)
        self.assertIn("task-2", active)
        self.assertNotIn("task-1", active)
        self.assertIn("alpha", monitor.metrics["agents"])
        self.assertIn("beta", monitor.metrics["agents"])

    def test_dispatch_entry_updates_task_state(self) -> None:
        from nucleus.tools import ohm

        monitor = ohm.OmegaHeartMonitor(bootstrap_tasks=False)
        entry = {
            "topic": "rd.tasks.dispatch",
            "ts": 100.0,
            "actor": "dispatcher",
            "data": {"req_id": "req-1", "target": "claude", "intent": "do thing"},
        }

        monitor._process_entry(entry)

        self.assertIn("req-1", monitor.task_state)
        task = monitor.task_state["req-1"]
        self.assertEqual(task["actor"], "claude")
        self.assertEqual(task["status"], "in_progress")
        self.assertEqual(task["desc"], "do thing")

    def test_pending_pairs_uses_oldest_age(self) -> None:
        from nucleus.tools import ohm

        monitor = ohm.OmegaHeartMonitor(bootstrap_tasks=False)
        entry = {
            "topic": "omega.pending.pairs",
            "ts": 123.0,
            "data": {"total": 5, "oldest_age_s": 12.5},
        }

        monitor._process_entry(entry)

        self.assertEqual(monitor.omega["pairs"]["total"], 5)
        self.assertEqual(monitor.omega["pairs"]["oldest_age_s"], 12.5)

    def test_health_and_provider_updates(self) -> None:
        from nucleus.tools import ohm

        monitor = ohm.OmegaHeartMonitor(bootstrap_tasks=False)
        monitor._process_entry(
            {"topic": "omega.health.composite", "ts": 1.0, "data": {"status": "degraded"}}
        )
        monitor._process_entry(
            {
                "topic": "omega.providers.health",
                "ts": 2.0,
                "data": {"providers": {"gemini": {"available": False}, "ollama": {"healthy": True}}},
            }
        )

        self.assertEqual(monitor.omega["health"]["status"], "degraded")
        provider_str = monitor._provider_string()
        self.assertIn("gemini:off", provider_str)
        self.assertIn("ollama:on", provider_str)

    def test_tail_task_ledger_reads_new_entries(self) -> None:
        from nucleus.tools import ohm

        monitor = ohm.OmegaHeartMonitor(bootstrap_tasks=False)
        with TemporaryDirectory() as td:
            path = Path(td) / "task_ledger.ndjson"
            entry1 = {"id": "a1", "run_id": "run-1", "status": "in_progress", "actor": "alpha"}
            path.write_text(json.dumps(entry1) + "\n", encoding="utf-8")
            monitor.task_ledger_path = str(path)

            monitor.tail_task_ledger()
            self.assertIn("run-1", monitor.task_state)

            entry2 = {"id": "a2", "run_id": "run-2", "status": "blocked", "actor": "beta"}
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry2) + "\n")

            monitor.tail_task_ledger()

        self.assertIn("run-2", monitor.task_state)

    def test_prune_tasks_removes_old_entries(self) -> None:
        from nucleus.tools import ohm

        monitor = ohm.OmegaHeartMonitor(bootstrap_tasks=False)
        now = time.time()
        monitor.task_stale_s = 10

        monitor.task_state["old_done"] = {
            "status": "completed",
            "completed_ts": now - 120,
            "last_ts": now - 120,
            "ts": now - 120,
        }
        monitor.task_state["old_stale"] = {
            "status": "in_progress",
            "last_ts": now - 120,
            "ts": now - 120,
        }
        monitor.task_state["fresh"] = {
            "status": "in_progress",
            "last_ts": now - 2,
            "ts": now - 2,
        }

        monitor._prune_tasks(now=now)

        self.assertNotIn("old_done", monitor.task_state)
        self.assertNotIn("old_stale", monitor.task_state)
        self.assertIn("fresh", monitor.task_state)

    def test_bus_size_mb_reads_file(self) -> None:
        from nucleus.tools import ohm

        monitor = ohm.OmegaHeartMonitor(bootstrap_tasks=False)
        with TemporaryDirectory() as td:
            path = Path(td) / "events.ndjson"
            path.write_bytes(b"x" * (1024 * 1024 * 2))
            size = monitor._bus_size_mb(path=str(path))

        self.assertGreaterEqual(size, 2.0)

    def test_status_payload_includes_a2a_counts(self) -> None:
        from nucleus.tools import ohm

        monitor = ohm.OmegaHeartMonitor(bootstrap_tasks=False)
        monitor.metrics["a2a_requests"] = 2
        monitor.metrics["a2a_responses"] = 1

        payload = monitor._status_payload()

        self.assertEqual(payload["a2a"]["requests"], 2)
        self.assertEqual(payload["a2a"]["responses"], 1)


if __name__ == "__main__":
    unittest.main()
