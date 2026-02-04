import importlib.util
import os
import pathlib
import sys
import tempfile
import unittest
import unittest.mock


class _FakePage:
    def __init__(self, *, url: str) -> None:
        self.url = url
        self.goto_urls: list[str] = []
        self._body_text = ""
        self._title = "Google AI Studio"

    async def goto(self, url: str, *, wait_until: str | None = None, timeout: int | None = None) -> None:
        self.goto_urls.append(url)
        self.url = url

    async def bring_to_front(self) -> None:
        return None

    async def title(self) -> str:
        return self._title

    async def text_content(self, selector: str) -> str:
        return self._body_text


class TestBrowserSessionDaemonAutoLoginCurrentUrlLower(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        tools_dir = pathlib.Path(__file__).resolve().parents[1]
        daemon_path = tools_dir / "browser_session_daemon.py"
        spec = importlib.util.spec_from_file_location("browser_session_daemon", daemon_path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)  # type: ignore[assignment]
        self.mod = mod

    async def test_auto_login_provider_does_not_crash_on_current_url_lower(self) -> None:
        async def _fake_google_login_flow(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            return {"success": True, "needs_code": False, "blocked_insecure": False, "message": "ok"}

        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            bus = root / ".pluribus" / "bus"
            bus.mkdir(parents=True, exist_ok=True)

            daemon = self.mod.BrowserSessionDaemon(root=root, bus_dir=bus)
            daemon.pages = {"gemini-web": _FakePage(url="https://aistudio.google.com/")}
            daemon.state.tabs = {
                "gemini-web": self.mod.TabSession(
                    provider_id="gemini-web",
                    tab_id="t1",
                    url="https://aistudio.google.com/",
                    status="needs_login",
                )
            }

            with unittest.mock.patch.dict(
                os.environ,
                {
                    "PLURIBUS_GOOGLE_USER": "u@example.com",
                    "PLURIBUS_GOOGLE_PASS": "pw",
                },
                clear=False,
            ):
                with unittest.mock.patch.object(self.mod, "_google_login_flow", _fake_google_login_flow):
                    with unittest.mock.patch.object(self.mod, "_looks_like_login_flow", lambda *_a, **_k: False):
                        res = await daemon._auto_login_provider("gemini-web")

        self.assertTrue(res.get("success"), res)
        self.assertNotEqual(res.get("status"), "error", res)


if __name__ == "__main__":
    unittest.main()

