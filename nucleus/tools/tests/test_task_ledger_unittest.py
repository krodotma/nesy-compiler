import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


class TestTaskLedger(unittest.TestCase):
    def test_append_and_read(self) -> None:
        from nucleus.tools.task_ledger import append_entry, read_entries, STATUS_VALUES

        with TemporaryDirectory() as td:
            ledger_path = Path(td) / "task_ledger.ndjson"
            entry = append_entry(
                {
                    "req_id": "req-1",
                    "actor": "tester",
                    "topic": "task.plan",
                    "status": "planned",
                    "intent": "test ledger append",
                },
                ledger_path=ledger_path,
                emit_bus=False,
            )

            self.assertIn(entry["status"], STATUS_VALUES)
            self.assertTrue(ledger_path.exists())

            entries = list(read_entries(ledger_path))
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["req_id"], "req-1")
            self.assertEqual(entries[0]["actor"], "tester")

    def test_filters(self) -> None:
        from nucleus.tools.task_ledger import append_entry, read_entries

        with TemporaryDirectory() as td:
            ledger_path = Path(td) / "task_ledger.ndjson"
            append_entry(
                {
                    "req_id": "req-1",
                    "actor": "tester",
                    "topic": "task.plan",
                    "status": "planned",
                },
                ledger_path=ledger_path,
                emit_bus=False,
            )
            append_entry(
                {
                    "req_id": "req-2",
                    "actor": "tester",
                    "topic": "task.exec",
                    "status": "completed",
                },
                ledger_path=ledger_path,
                emit_bus=False,
            )

            planned = list(read_entries(ledger_path, status="planned"))
            self.assertEqual(len(planned), 1)
            self.assertEqual(planned[0]["req_id"], "req-1")

            exec_entries = list(read_entries(ledger_path, topic="task.exec"))
            self.assertEqual(len(exec_entries), 1)
            self.assertEqual(exec_entries[0]["req_id"], "req-2")

    def test_schema_rejects_missing_fields(self) -> None:
        from nucleus.tools.task_ledger import append_entry

        with TemporaryDirectory() as td:
            ledger_path = Path(td) / "task_ledger.ndjson"
            with self.assertRaises(ValueError):
                append_entry(
                    {"actor": "tester", "topic": "task.plan", "status": "planned"},
                    ledger_path=ledger_path,
                    emit_bus=False,
                )

    def test_json_round_trip(self) -> None:
        from nucleus.tools.task_ledger import append_entry

        with TemporaryDirectory() as td:
            ledger_path = Path(td) / "task_ledger.ndjson"
            entry = append_entry(
                {
                    "req_id": "req-3",
                    "actor": "tester",
                    "topic": "task.plan",
                    "status": "planned",
                    "meta": {"priority": "high"},
                },
                ledger_path=ledger_path,
                emit_bus=False,
            )

            raw = ledger_path.read_text(encoding="utf-8").strip()
            decoded = json.loads(raw)
            self.assertEqual(decoded["id"], entry["id"])

    def test_emit_bus_event(self) -> None:
        from nucleus.tools import agent_bus
        from nucleus.tools.task_ledger import append_entry

        with TemporaryDirectory() as td:
            ledger_path = Path(td) / "task_ledger.ndjson"
            dummy_paths = agent_bus.BusPaths(
                active_dir=td,
                events_path=str(Path(td) / "events.ndjson"),
                primary_dir=td,
                fallback_dir=None,
            )
            with patch("nucleus.tools.task_ledger.agent_bus.resolve_bus_paths", return_value=dummy_paths), \
                patch("nucleus.tools.task_ledger.agent_bus.emit_event") as emit:
                entry = append_entry(
                    {
                        "req_id": "req-4",
                        "actor": "tester",
                        "topic": "task.plan",
                        "status": "planned",
                    },
                    ledger_path=ledger_path,
                    emit_bus=True,
                )

                self.assertEqual(entry["req_id"], "req-4")
                self.assertTrue(emit.called)
                kwargs = emit.call_args.kwargs
                self.assertEqual(kwargs["topic"], "task_ledger.append")


if __name__ == "__main__":
    unittest.main()
