import json
import pathlib
import tempfile
import unittest

import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SDK_DIR = REPO_ROOT / "nucleus" / "sdk"
TOOLS_DIR = REPO_ROOT / "nucleus" / "tools"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SDK_DIR))
sys.path.insert(0, str(TOOLS_DIR))

from nucleus.sdk.flow import load_flow  # noqa: E402


class TestSdkFlowRoundtrip(unittest.TestCase):
    def test_safe_py_flow_load_and_roundtrip(self):
        with tempfile.TemporaryDirectory(prefix="pluribus_flow_") as td:
            d = pathlib.Path(td)
            src = d / "example_flow.py"
            src.write_text(
                "\n".join(
                    [
                        "from nucleus.sdk.flow import Agent, Merge",
                        "",
                        "root = Agent(role='Researcher', goal='Find papers', tools=['rg'])",
                        "branch_a = root.fork(role='Critic', goal='Critique', tools=['pytest'])",
                        "branch_b = root.fork(role='Fan')",
                        "synth = Merge([branch_a, branch_b], strategy='Consensus')",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            f1 = load_flow(src)
            self.assertEqual(len(f1.nodes), 4)
            self.assertEqual(len(f1.edges), 4)

            out = d / "roundtrip.py"
            f1.save_py(out)
            f2 = load_flow(out)
            c1 = f1.canonical()
            c2 = f2.canonical()
            c2["name"] = c1.get("name", c2.get("name"))
            self.assertEqual(c1, c2)

    def test_roundtrip_tool_emits_bus_metric(self):
        tool = TOOLS_DIR / "studio_flow_roundtrip.py"
        self.assertTrue(tool.exists())

        with tempfile.TemporaryDirectory(prefix="pluribus_flow_") as td:
            d = pathlib.Path(td)
            bus_dir = d / "bus"
            bus_dir.mkdir(parents=True, exist_ok=True)
            (bus_dir / "events.ndjson").write_text("", encoding="utf-8")

            src = d / "example_flow.py"
            src.write_text(
                "\n".join(
                    [
                        "from nucleus.sdk.flow import Agent, Merge",
                        "root = Agent(role='Researcher', goal='Find papers')",
                        "branch_a = root.fork(role='Critic')",
                        "branch_b = root.fork(role='Fan')",
                        "synth = Merge([branch_a, branch_b])",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            import subprocess

            p = subprocess.run(
                [
                    sys.executable,
                    str(tool),
                    "--root",
                    str(d),
                    "--bus-dir",
                    str(bus_dir),
                    "--actor",
                    "tester",
                    "--in",
                    str(src),
                    "--emit-bus",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=10.0,
                env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            self.assertEqual(p.returncode, 0, msg=(p.stderr or p.stdout))
            events = (bus_dir / "events.ndjson").read_text(encoding="utf-8", errors="replace").splitlines()
            found = False
            for line in events:
                if not line.strip():
                    continue
                e = json.loads(line)
                if e.get("topic") == "studio.flow.roundtrip" and e.get("kind") == "metric":
                    found = True
                    self.assertTrue(e.get("data", {}).get("ok"))
                    break
            self.assertTrue(found)


if __name__ == "__main__":
    unittest.main()
