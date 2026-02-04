import sys
import unittest
import io
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pbpair  # noqa: E402


class TestPBPair(unittest.TestCase):
    def test_build_prompt_contains_contract_and_statusline(self):
        ctx = pbpair.ContextPack(mode="min", root=None, rhizome=None, status=None, includes=[])
        p = pbpair.build_pbpair_prompt(role="planner", provider="mock", question="hello", mode="min", ctx=ctx)
        self.assertIn("PBPAIR request", p)
        self.assertIn("ROLE: planner", p)
        self.assertIn("PROVIDER: mock", p)
        self.assertIn("QUESTION:", p)
        self.assertIn("STATUSLINE:", p)
        self.assertIn("\"gaps\"", p)

    def test_context_alias_and_quiet_flag(self):
        parser = pbpair.build_parser()
        args = parser.parse_args(
            [
                "--context",
                "min",
                "--quiet",
                "--provider",
                "mock",
                "--flow",
                "m",
                "--prompt",
                "hi",
                "--bus-dir",
                "/tmp/does-not-matter",
                "--root",
                "/tmp/does-not-matter",
            ]
        )
        self.assertEqual(args.mode, "min")
        self.assertTrue(args.quiet)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "root"
            bus = Path(td) / "bus"
            root.mkdir(parents=True, exist_ok=True)
            bus.mkdir(parents=True, exist_ok=True)

            out = io.StringIO()
            with redirect_stdout(out):
                rc = pbpair.main(
                    [
                        "--bus-dir",
                        str(bus),
                        "--root",
                        str(root),
                        "--provider",
                        "mock",
                        "--flow",
                        "m",
                        "--context",
                        "min",
                        "--prompt",
                        "ok",
                        "--quiet",
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertEqual(out.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
