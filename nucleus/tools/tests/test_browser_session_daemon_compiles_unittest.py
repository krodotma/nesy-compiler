import py_compile
import tempfile
import unittest
from pathlib import Path


class TestBrowserSessionDaemonCompiles(unittest.TestCase):
    def test_browser_session_daemon_compiles(self) -> None:
        tools_dir = Path(__file__).resolve().parents[1]
        target = tools_dir / "browser_session_daemon.py"
        with tempfile.TemporaryDirectory(prefix="pluribus_compile_") as td:
            cfile = Path(td) / "browser_session_daemon.pyc"
            py_compile.compile(str(target), cfile=str(cfile), doraise=True)
