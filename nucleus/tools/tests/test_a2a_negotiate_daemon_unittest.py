import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


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


class TestA2ANegotiateDaemon(unittest.TestCase):
    def test_negotiate_daemon_agrees_when_capabilities_match(self):
        tool = pathlib.Path(__file__).resolve().parents[1] / "a2a" / "negotiate_daemon.py"
        self.assertTrue(tool.exists())

        with tempfile.TemporaryDirectory(prefix="pluribus_bus_") as td:
            bus_dir = pathlib.Path(td)
            req_id = "r1"
            request_payload = {
                "req_id": req_id,
                "initiator": "initiator",
                "target": "agent-x",
                "task_description": "Do the thing",
                "constraints": {"required_capabilities": ["python"]},
                "compensation": {"max_tokens": 5000},
            }
            bus_tool = pathlib.Path(__file__).resolve().parents[1] / "agent_bus.py"
            subprocess.run(
                [
                    sys.executable,
                    str(bus_tool),
                    "--bus-dir",
                    str(bus_dir),
                    "pub",
                    "--topic",
                    "a2a.negotiate.request",
                    "--kind",
                    "request",
                    "--level",
                    "info",
                    "--actor",
                    "initiator",
                    "--data",
                    json.dumps(request_payload, ensure_ascii=False),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            p = subprocess.run(
                [sys.executable, str(tool), "--bus-dir", str(bus_dir), "--actor", "agent-x", "--capabilities", "python,git", "--once"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10.0,
            )
            self.assertEqual(p.returncode, 0, msg=(p.stderr or p.stdout))

            events = _read_events(bus_dir)
            resp = None
            for e in events:
                if e.get("topic") == "a2a.negotiate.response" and e.get("kind") == "response":
                    d = e.get("data")
                    if isinstance(d, dict) and d.get("req_id") == req_id:
                        resp = d
                        break
            self.assertIsNotNone(resp)
            self.assertEqual(resp.get("decision"), "agree")

    def test_negotiate_daemon_rejects_when_missing_capabilities(self):
        tool = pathlib.Path(__file__).resolve().parents[1] / "a2a" / "negotiate_daemon.py"
        bus_tool = pathlib.Path(__file__).resolve().parents[1] / "agent_bus.py"

        with tempfile.TemporaryDirectory(prefix="pluribus_bus_") as td:
            bus_dir = pathlib.Path(td)
            req_id = "r2"
            request_payload = {
                "req_id": req_id,
                "initiator": "initiator",
                "target": "agent-y",
                "task_description": "Need rust",
                "constraints": {"required_capabilities": ["rust"]},
                "compensation": {"max_tokens": 5000},
            }
            subprocess.run(
                [
                    sys.executable,
                    str(bus_tool),
                    "--bus-dir",
                    str(bus_dir),
                    "pub",
                    "--topic",
                    "a2a.negotiate.request",
                    "--kind",
                    "request",
                    "--level",
                    "info",
                    "--actor",
                    "initiator",
                    "--data",
                    json.dumps(request_payload, ensure_ascii=False),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            p = subprocess.run(
                [sys.executable, str(tool), "--bus-dir", str(bus_dir), "--actor", "agent-y", "--capabilities", "python", "--once"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10.0,
            )
            self.assertEqual(p.returncode, 0, msg=(p.stderr or p.stdout))

            events = _read_events(bus_dir)
            decline = None
            redirect = None
            for e in events:
                if e.get("topic") == "a2a.decline" and e.get("kind") == "response":
                    d = e.get("data")
                    if isinstance(d, dict) and d.get("req_id") == req_id:
                        decline = d
                        break
            self.assertIsNotNone(decline)

            for e in events:
                if e.get("topic") == "a2a.redirect" and e.get("kind") == "response":
                    d = e.get("data")
                    if isinstance(d, dict) and d.get("req_id") == req_id:
                        redirect = d
                        break
            self.assertIsNone(redirect)

    def test_negotiate_daemon_emits_redirect_when_alternative_provided(self):
        tool = pathlib.Path(__file__).resolve().parents[1] / "a2a" / "negotiate_daemon.py"
        bus_tool = pathlib.Path(__file__).resolve().parents[1] / "agent_bus.py"

        with tempfile.TemporaryDirectory(prefix="pluribus_bus_") as td:
            bus_dir = pathlib.Path(td)
            req_id = "r3"
            request_payload = {
                "req_id": req_id,
                "initiator": "initiator",
                "target": "agent-z",
                "task_description": "Need rust",
                "constraints": {"required_capabilities": ["rust"]},
                "compensation": {"max_tokens": 5000},
                "alternatives": ["agent-rust"],
            }
            subprocess.run(
                [
                    sys.executable,
                    str(bus_tool),
                    "--bus-dir",
                    str(bus_dir),
                    "pub",
                    "--topic",
                    "a2a.negotiate.request",
                    "--kind",
                    "request",
                    "--level",
                    "info",
                    "--actor",
                    "initiator",
                    "--data",
                    json.dumps(request_payload, ensure_ascii=False),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            p = subprocess.run(
                [sys.executable, str(tool), "--bus-dir", str(bus_dir), "--actor", "agent-z", "--capabilities", "python", "--once"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10.0,
            )
            self.assertEqual(p.returncode, 0, msg=(p.stderr or p.stdout))

            events = _read_events(bus_dir)
            redirect = None
            for e in events:
                if e.get("topic") == "a2a.redirect" and e.get("kind") == "response":
                    d = e.get("data")
                    if isinstance(d, dict) and d.get("req_id") == req_id:
                        redirect = d
                        break
            self.assertIsNotNone(redirect)
            self.assertEqual(redirect.get("redirect_to"), "agent-rust")


if __name__ == "__main__":
    unittest.main()
