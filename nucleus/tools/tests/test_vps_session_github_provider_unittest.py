import json
import tempfile
import unittest
from pathlib import Path

import sys
import pathlib

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from vps_session import VPSSessionManager  # noqa: E402


class TestVPSSessionGitHubProvider(unittest.TestCase):
    def test_default_session_includes_github_provider(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".pluribus").mkdir(parents=True, exist_ok=True)
            (root / ".pluribus" / "rhizome.json").write_text("{}", encoding="utf-8")

            mgr = VPSSessionManager(root)
            session = mgr.load()
            self.assertTrue({"chatgpt-web", "claude-web", "gemini-web"}.issubset(set(session.providers.keys())))
            self.assertNotIn("github", session.providers)

            mgr.save()
            obj = json.loads((root / ".pluribus" / "vps_session.json").read_text(encoding="utf-8"))
            self.assertTrue({"chatgpt-web", "claude-web", "gemini-web"}.issubset(set((obj.get("providers", {}) or {}).keys())))
            self.assertNotIn("github", obj.get("providers", {}))


if __name__ == "__main__":
    unittest.main()
