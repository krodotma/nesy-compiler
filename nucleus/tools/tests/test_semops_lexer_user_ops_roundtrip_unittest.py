import json
import tempfile
import unittest
from pathlib import Path


class TestSemopsLexerUserOpsRoundtrip(unittest.TestCase):
    def test_define_persists_and_reload_preserves_fields(self) -> None:
        from nucleus.tools.semops_lexer import SemopsLexer  # namespace import

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            semops_path = root / "semops.json"
            user_ops_path = root / "user_operators.json"

            semops_path.write_text(
                json.dumps(
                    {
                        "operators": {
                            "CKIN": {
                                "id": "ckin",
                                "name": "CKIN",
                                "domain": "observability",
                                "category": "status",
                                "description": "check-in",
                                "aliases": ["ckin", "CKIN"],
                            }
                        },
                        "grammar": {"slash_command_pattern": "^/(help|ckin)(\\s+.*)?$"},
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            lexer = SemopsLexer(semops_path=semops_path, user_ops_path=user_ops_path)
            lexer.define_operator(
                "MYOP",
                op_id="myop",
                name="MYOP",
                description="my op",
                aliases=["myop", "MYOP", "checking in"],
                domain="user",
                category="custom",
                tool="nucleus/tools/myop_operator.py",
                bus_topic="operator.myop.request",
                bus_kind="request",
                secondary_topic="operator.myop.ack",
                ui={"route": "/#semops", "component": "SemopsEditor"},
                agents=["codex", "gemini"],
                apps=["dashboard"],
                targets=[{"type": "tool", "ref": "nucleus/tools/myop_operator.py"}],
                extra={"x_custom": {"foo": 1}},
                persist=True,
            )

            raw = json.loads(user_ops_path.read_text(encoding="utf-8"))
            self.assertEqual(raw.get("schema_version"), 1)
            self.assertIn("operators", raw)
            self.assertIn("MYOP", raw["operators"])
            self.assertEqual(raw["operators"]["MYOP"]["bus_kind"], "request")
            self.assertIn("x_custom", raw["operators"]["MYOP"])

            # Reload should preserve fields.
            lexer2 = SemopsLexer(semops_path=semops_path, user_ops_path=user_ops_path)
            self.assertIn("MYOP", lexer2.operators)
            op = lexer2.operators["MYOP"]
            self.assertTrue(op.user_defined)
            self.assertEqual(op.bus_kind, "request")
            self.assertEqual(op.secondary_topic, "operator.myop.ack")
            self.assertEqual(op.ui.get("component"), "SemopsEditor")
            self.assertIn("checking in", op.aliases)
            self.assertIn("x_custom", op.extra)


if __name__ == "__main__":
    unittest.main()

