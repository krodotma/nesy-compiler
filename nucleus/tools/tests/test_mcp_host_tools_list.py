import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestMcpHostToolsList(unittest.TestCase):
    def test_host_tools_lists_tools(self) -> None:
        root = Path(__file__).resolve().parents[2]
        host = root / "mcp" / "host.py"

        with tempfile.TemporaryDirectory() as td:
            rh = Path(td)
            (rh / ".pluribus").mkdir(parents=True, exist_ok=True)
            (rh / ".pluribus" / "rhizome.json").write_text("{}", encoding="utf-8")

            env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PLURIBUS_ROOT": str(rh)}
            p = subprocess.run(
                [os.environ.get("PYTHON", "python3"), str(host), "--root", str(rh), "tools", "--server", "pluribus-rhizome"],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(p.returncode, 0, p.stderr)
            obj = json.loads(p.stdout)
            tools = [t.get("name") for t in (obj.get("tools") or []) if isinstance(t, dict)]
            self.assertIn("ingest", tools)
            self.assertIn("list_artifacts", tools)
            self.assertIn("show_artifact", tools)


if __name__ == "__main__":
    unittest.main()

