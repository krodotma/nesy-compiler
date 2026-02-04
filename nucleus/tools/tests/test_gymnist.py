import json
import pathlib
import sys
import tempfile
import unittest

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import gymnist  # noqa: E402


class TestGymnist(unittest.TestCase):
    def test_emits_dialogos_submit(self):
        with tempfile.TemporaryDirectory() as td:
            bus_dir = pathlib.Path(td) / "bus"
            bus_dir.mkdir(parents=True, exist_ok=True)
            (bus_dir / "events.ndjson").write_text("", encoding="utf-8")

            # Emit a request directly (bypassing CLI) and verify it is in the bus file.
            req_id = "r1"
            gymnist.emit_bus(
                bus_dir,
                topic="dialogos.submit",
                kind="request",
                level="info",
                actor="gymnist-test",
                data={"req_id": req_id, "mode": "llm", "providers": ["mock"], "prompt": "hi"},
            )

            lines = [json.loads(l) for l in (bus_dir / "events.ndjson").read_text(encoding="utf-8").splitlines() if l.strip()]
            submits = [l for l in lines if l.get("topic") == "dialogos.submit" and (l.get("data") or {}).get("req_id") == req_id]
            self.assertEqual(len(submits), 1)

    def test_format_repl_prompt_includes_history(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        p = gymnist.format_repl_prompt(
            user_prompt="what next?",
            history=history,
            system="be brief",
        )
        self.assertIn("System: be brief", p)
        self.assertIn("User: hi", p)
        self.assertIn("Assistant: hello", p)
        self.assertIn("User: what next?", p)

    def test_collect_text_from_events(self):
        req_id = "r1"
        events = [
            {"topic": "dialogos.cell.start", "data": {"req_id": req_id}},
            {"topic": "dialogos.cell.output", "data": {"req_id": req_id, "content": "a"}},
            {"topic": "dialogos.cell.output", "data": {"req_id": req_id, "content": "b"}},
            {"topic": "dialogos.cell.end", "data": {"req_id": req_id}},
        ]
        self.assertEqual(gymnist.collect_text_for_req_id(events=events, req_id=req_id), "a\nb")

    def test_append_repl_session_event_is_ndjson(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td) / "root"
            root.mkdir(parents=True, exist_ok=True)
            session_path = gymnist.append_repl_session_event(
                root=root,
                session_id="s1",
                req_id="r1",
                role="user",
                content="hello",
                provider="mock",
                ok=True,
            )
            self.assertTrue(session_path.exists())
            lines = [json.loads(l) for l in session_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertEqual(len(lines), 1)
            self.assertEqual(lines[0].get("session_id"), "s1")
            self.assertEqual(lines[0].get("req_id"), "r1")
            self.assertEqual(lines[0].get("role"), "user")
            self.assertEqual(lines[0].get("content"), "hello")


if __name__ == "__main__":
    unittest.main()
