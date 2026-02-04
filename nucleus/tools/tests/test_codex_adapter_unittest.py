import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from codex_adapter import CodexAdapter, _read_package_version


class TestCodexAdapter(unittest.TestCase):
    def test_read_package_version(self) -> None:
        with tempfile.TemporaryDirectory(prefix="codex_adapter_") as td:
            package_json = Path(td) / "package.json"
            package_json.write_text(json.dumps({"version": "0.75.0"}), encoding="utf-8")
            self.assertEqual(_read_package_version(package_json), "0.75.0")

    def test_status_uses_local_package(self) -> None:
        with tempfile.TemporaryDirectory(prefix="codex_adapter_") as td:
            root = Path(td)
            package_dir = root / "node_modules" / "@openai" / "codex"
            package_dir.mkdir(parents=True, exist_ok=True)
            (package_dir / "package.json").write_text(json.dumps({"version": "0.75.0"}), encoding="utf-8")

            adapter = CodexAdapter(bus_dir=str(root / "bus"), repo_root=root)
            with mock.patch("codex_adapter.shutil.which", return_value=None):
                status = adapter.status()

            self.assertTrue(status.ok)
            self.assertEqual(status.npm_version, "0.75.0")
            self.assertIsNone(status.cli_path)

    def test_status_missing(self) -> None:
        with tempfile.TemporaryDirectory(prefix="codex_adapter_") as td:
            root = Path(td)
            adapter = CodexAdapter(bus_dir=str(root / "bus"), repo_root=root)
            with mock.patch("codex_adapter.shutil.which", return_value=None):
                status = adapter.status()

            self.assertFalse(status.ok)
            self.assertIsNotNone(status.notes)
