import json
import pathlib
import tempfile
import unittest

import sys

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import plurichat  # noqa: E402


def _read_events(bus_dir: pathlib.Path) -> list[dict]:
    p = bus_dir / "events.ndjson"
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        out.append(json.loads(line))
    return out


class TestPluriChatRealagents(unittest.TestCase):
    def test_handle_command_realagents_emits_rd_tasks_dispatch(self):
        with tempfile.TemporaryDirectory(prefix="pluribus_bus_") as td:
            bus_dir = pathlib.Path(td)
            (bus_dir / "events.ndjson").write_text("", encoding="utf-8")

            state = plurichat.ChatState(bus_dir=bus_dir, actor="tester")
            ok = plurichat.handle_command("/realagents targets=codex task_id=REALAGENTS_upgrade Deepen MCP inventory emission", state)
            self.assertTrue(ok)

            events = _read_events(bus_dir)
            found = any(e.get("topic") == "rd.tasks.dispatch" and e.get("kind") == "request" for e in events)
            self.assertTrue(found)


if __name__ == "__main__":
    unittest.main()

