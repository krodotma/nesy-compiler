import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestDomainIntent(unittest.TestCase):
    def test_apply_then_validate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".pluribus" / "index").mkdir(parents=True, exist_ok=True)
            (root / ".pluribus" / "rhizome.json").write_text(
                json.dumps({"schema_version": 1, "name": "t", "domains": ["avtr.you", "avtr.world"]}, ensure_ascii=False),
                encoding="utf-8",
            )
            (root / "nucleus" / "docs" / "ingresses").mkdir(parents=True, exist_ok=True)
            intent_path = root / "nucleus" / "docs" / "ingresses" / "domain_intent.json"
            intent_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "domains": [
                            {"domain": "avtr.you", "purpose": "product", "tags": ["product", "avtr"]},
                            {"domain": "example.com", "purpose": "portfolio", "tags": ["portfolio", "ingress"]},
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            cmd_apply = ["python3", "/pluribus/nucleus/tools/domain_intent.py", "--root", str(root), "apply"]
            p_apply = subprocess.run(cmd_apply, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertEqual(p_apply.returncode, 0, p_apply.stderr)

            cmd_validate = ["python3", "/pluribus/nucleus/tools/domain_intent.py", "--root", str(root), "validate"]
            p_val = subprocess.run(cmd_validate, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertEqual(p_val.returncode, 0, p_val.stderr + "\n" + p_val.stdout)


if __name__ == "__main__":
    unittest.main()

