import importlib.util
import pathlib
import sys
import unittest
import unittest.mock


class TestBrowserSessionDaemonTotpPythonFallback(unittest.TestCase):
    def test_generate_totp_falls_back_without_oathtool(self) -> None:
        tools_dir = pathlib.Path(__file__).resolve().parents[1]
        daemon_path = tools_dir / "browser_session_daemon.py"
        spec = importlib.util.spec_from_file_location("browser_session_daemon", daemon_path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)  # type: ignore[assignment]

        # RFC 6238 test secret (ASCII "12345678901234567890") base32-encoded.
        secret = "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ"

        with unittest.mock.patch.object(mod.subprocess, "run", side_effect=FileNotFoundError()):
            with unittest.mock.patch.object(mod.time, "time", return_value=59):
                code = mod._generate_totp(secret)  # type: ignore[attr-defined]

        self.assertEqual(code, "287082")


if __name__ == "__main__":
    unittest.main()

