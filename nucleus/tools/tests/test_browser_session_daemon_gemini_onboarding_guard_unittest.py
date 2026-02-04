import importlib.util
import os
import pathlib
import sys
import tempfile
import unittest
import unittest.mock


class _FakePage:
    def __init__(self, *, url: str, click_redirect_url: str | None, raise_on_click: bool) -> None:
        self.url = url
        self._click_redirect_url = click_redirect_url
        self._raise_on_click = raise_on_click
        self._title = "Google AI Studio"
        self._body_text = ""
        self.click_selectors: list[str] = []

    async def bring_to_front(self) -> None:
        return None

    async def title(self) -> str:
        return self._title

    async def text_content(self, selector: str) -> str:
        return self._body_text

    async def click(self, selector: str, *, timeout: int | None = None) -> None:
        self.click_selectors.append(selector)
        if self._raise_on_click:
            raise TimeoutError("click failed")
        if self._click_redirect_url:
            self.url = self._click_redirect_url
            self._click_redirect_url = None

    async def wait_for_load_state(self, state: str, *, timeout: int | None = None) -> None:
        return None


class TestBrowserSessionDaemonGeminiOnboardingGuard(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        tools_dir = pathlib.Path(__file__).resolve().parents[1]
        daemon_path = tools_dir / "browser_session_daemon.py"
        spec = importlib.util.spec_from_file_location("browser_session_daemon", daemon_path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)  # type: ignore[assignment]
        self.mod = mod

    async def test_auto_login_returns_needs_onboarding_when_stuck_on_welcome(self) -> None:
        async def _fake_google_login_flow(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            return {"success": True, "needs_code": False, "blocked_insecure": False, "message": "ok"}

        async def _sleep_noop(*_args, **_kwargs) -> None:
            return None

        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            bus = root / ".pluribus" / "bus"
            bus.mkdir(parents=True, exist_ok=True)

            daemon = self.mod.BrowserSessionDaemon(root=root, bus_dir=bus)
            daemon.pages = {
                "gemini-web": _FakePage(
                    url="https://aistudio.google.com/welcome",
                    click_redirect_url=None,
                    raise_on_click=True,
                )
            }
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
                {"PLURIBUS_GOOGLE_USER": "u@example.com", "PLURIBUS_GOOGLE_PASS": "pw"},
                clear=False,
            ):
                with unittest.mock.patch.object(self.mod, "_google_login_flow", _fake_google_login_flow):
                    with unittest.mock.patch.object(self.mod.asyncio, "sleep", _sleep_noop):
                        res = await daemon._auto_login_provider("gemini-web")

        self.assertEqual(res.get("status"), "needs_onboarding", res)
        self.assertFalse(res.get("success"), res)

    async def test_auto_login_does_not_mark_ready_when_onboarding_redirects_to_login(self) -> None:
        async def _fake_google_login_flow(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            return {"success": True, "needs_code": False, "blocked_insecure": False, "message": "ok"}

        async def _sleep_noop(*_args, **_kwargs) -> None:
            return None

        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            bus = root / ".pluribus" / "bus"
            bus.mkdir(parents=True, exist_ok=True)

            daemon = self.mod.BrowserSessionDaemon(root=root, bus_dir=bus)
            daemon.pages = {
                "gemini-web": _FakePage(
                    url="https://aistudio.google.com/welcome",
                    click_redirect_url="https://accounts.google.com/v3/signin/identifier",
                    raise_on_click=False,
                )
            }
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
                {"PLURIBUS_GOOGLE_USER": "u@example.com", "PLURIBUS_GOOGLE_PASS": "pw"},
                clear=False,
            ):
                with unittest.mock.patch.object(self.mod, "_google_login_flow", _fake_google_login_flow):
                    with unittest.mock.patch.object(self.mod.asyncio, "sleep", _sleep_noop):
                        res = await daemon._auto_login_provider("gemini-web")

        self.assertEqual(res.get("status"), "needs_login", res)
        self.assertFalse(res.get("success"), res)


if __name__ == "__main__":
    unittest.main()

