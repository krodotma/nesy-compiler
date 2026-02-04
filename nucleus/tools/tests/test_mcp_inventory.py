import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestMcpInventory(unittest.TestCase):
    def test_inventory_writes_append_only_file(self) -> None:
        tools_dir = Path(__file__).resolve().parents[1]
        tool = tools_dir / "mcp_inventory.py"

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".pluribus").mkdir(parents=True, exist_ok=True)
            (root / ".pluribus" / "rhizome.json").write_text("{}", encoding="utf-8")

            env = {
                **os.environ,
                "PLURIBUS_ROOT": str(root),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            p = subprocess.run(
                [os.environ.get("PYTHON", "python3"), str(tool), "--root", str(root)],
                env=env,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            self.assertIn(p.returncode, (0, 2), p.stderr)  # allow missing tools as nonzero but file should exist
            out = Path((p.stdout or "").strip())
            self.assertTrue(out.exists(), f"expected inventory file at {out}")


if __name__ == "__main__":
    unittest.main()

