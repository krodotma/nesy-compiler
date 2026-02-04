import asyncio
import importlib.util
import os
import pathlib
import sys
import tempfile
import unittest
import unittest.mock
import uuid


class _FakeKeyboard:
    def __init__(self) -> None:
        self.pressed: list[str] = []
        self.typed: list[tuple[str, int]] = []

    async def press(self, key: str) -> None:
        self.pressed.append(key)

    async def type(self, text: str, delay: int = 0) -> None:
        self.typed.append((text, delay))


class _FakeMouse:
    def __init__(self) -> None:
        self.clicked: list[tuple[int, int]] = []

    async def click(self, x: int, y: int) -> None:
        self.clicked.append((x, y))


class _FakePage:
    def __init__(self) -> None:
        self.filled: list[tuple[str, str]] = []
        self.keyboard = _FakeKeyboard()
        self.mouse = _FakeMouse()

    async def fill(self, selector: str, value: str) -> None:
        self.filled.append((selector, value))

    async def click(self, selector: str, timeout: int | None = None) -> None:
        return None

    async def goto(self, url: str, *, wait_until: str | None = None, timeout: int | None = None) -> None:
        return None


class TestBrowserSessionDaemonSolveChallengeBusTail(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        tools_dir = pathlib.Path(__file__).resolve().parents[1]
        daemon_path = tools_dir / "browser_session_daemon.py"
        spec = importlib.util.spec_from_file_location("browser_session_daemon", daemon_path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)  # type: ignore[assignment]
        self.mod = mod

    async def test_solve_challenge_tails_bus_and_applies_matching_response(self) -> None:
        fixed_uuid = uuid.UUID("00000000-0000-0000-0000-000000000000")
        fixed_req_id = str(fixed_uuid)

        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            bus = root / ".pluribus" / "bus"
            bus.mkdir(parents=True, exist_ok=True)

            with unittest.mock.patch.dict(
                os.environ,
                {
                    "PLURIBUS_BUS_DIR": str(bus),
                    "PLURIBUS_CHALLENGE_WAIT_S": "3",
                    "PLURIBUS_CHALLENGE_POLL_S": "0.05",
                    "PLURIBUS_CHALLENGE_MAX_READ_BYTES": "65536",
                },
                clear=False,
            ):
                daemon = self.mod.BrowserSessionDaemon(root=root, bus_dir=bus)
                # Keep this unit test independent of agent_bus mirroring behavior.
                self.mod.agent_bus = None
                self.mod.SOLVER_ENABLED = False
                self.mod.SOLVER_CMD = ""

                html_path = root / "snapshot.html"
                html_path.write_text("<html><body>otp</body></html>", encoding="utf-8")
                screenshot_path = root / "snapshot.png"
                screenshot_path.write_bytes(b"")

                async def _fake_snapshot(
                    _self: object,
                    _page: object,
                    _provider_id: str,
                    _tag: str,
                    _error_msg: str | None = None,
                ) -> dict:
                    return {"html": str(html_path), "screenshot": str(screenshot_path), "url": "about:blank", "title": "x"}

                daemon.save_snapshot = _fake_snapshot.__get__(daemon, self.mod.BrowserSessionDaemon)  # type: ignore[method-assign]

                page = _FakePage()

                with unittest.mock.patch.object(self.mod.uuid, "uuid4", return_value=fixed_uuid):
                    task = asyncio.create_task(daemon.solve_challenge(page, "gemini-web", reason="otp"))
                    await asyncio.sleep(0.1)
                    self.mod.append_bus_event(
                        bus,
                        "browser.challenge.response",
                        {"req_id": fixed_req_id, "provider": "gemini-web", "answer": "123456"},
                        actor="test",
                    )
                    res = await task

        self.assertEqual(res.get("req_id"), fixed_req_id)
        self.assertEqual(res.get("answer"), "123456")
        self.assertTrue(res.get("applied"))
        self.assertTrue(any(v == "123456" for _sel, v in page.filled))
        self.assertIn("Enter", page.keyboard.pressed)


if __name__ == "__main__":
    unittest.main()
