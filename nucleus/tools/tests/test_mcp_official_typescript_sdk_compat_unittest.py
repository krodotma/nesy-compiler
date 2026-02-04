import subprocess
import unittest


class TestMcpOfficialTypescriptSdkCompat(unittest.TestCase):
    def test_typescript_sdk_can_talk_to_py_servers(self):
        p = subprocess.run(
            ["node", "/pluribus/nucleus/mcp/compat/mcp_typescript_sdk_smoke.mjs"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if p.returncode != 0:
            raise AssertionError((p.stderr or p.stdout or "").strip() or f"exit={p.returncode}")
        self.assertIn("ok", (p.stdout or ""))


if __name__ == "__main__":
    unittest.main()

