import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class TestAgent0Adapter(unittest.TestCase):
    def test_create_plan_writes_entries(self) -> None:
        from nucleus.tools import agent0_adapter

        with TemporaryDirectory() as td:
            ledger_path = Path(td) / "task_ledger.ndjson"
            plan = agent0_adapter.create_plan(
                goal="test goal",
                iterations=2,
                actor="tester",
                req_id="req-1",
                run_id="run-1",
                ledger_path=ledger_path,
                emit_bus=False,
            )
            self.assertEqual(len(plan["entries"]), 4)
            self.assertTrue(ledger_path.exists())

    def test_check_agent0_install_missing(self) -> None:
        from nucleus.tools import agent0_adapter

        ok, detail = agent0_adapter.check_agent0_install(None)
        self.assertFalse(ok)
        self.assertEqual(detail, "agent0_root_missing")


if __name__ == "__main__":
    unittest.main()
