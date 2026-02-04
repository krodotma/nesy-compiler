import json
import pathlib
import sys
import unittest

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
ROOT = TOOLS_DIR.parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import persona_registry  # noqa: E402


class TestPersonaRegistry(unittest.TestCase):
    def test_persona_registry_validates_spec_file(self):
        path = ROOT / "nucleus" / "specs" / "personas.json"
        obj = json.loads(path.read_text(encoding="utf-8"))
        ok, errs = persona_registry.validate_registry(obj)
        self.assertTrue(ok, errs)

    def test_persona_choose_deep_audit_prefers_ring0_auditor(self):
        path = ROOT / "nucleus" / "specs" / "personas.json"
        reg = json.loads(path.read_text(encoding="utf-8"))
        pick = persona_registry.choose_persona(
            reg,
            goal="Audit PQC posture and security of provider router",
            depth="deep",
            kind="audit",
            effects="file",
        )
        self.assertIn(pick.persona_id, {"ring0.security_auditor", "ring0.architect"})

    def test_persona_choose_narrow_defaults_to_subagent_narrow_coder(self):
        path = ROOT / "nucleus" / "specs" / "personas.json"
        reg = json.loads(path.read_text(encoding="utf-8"))
        pick = persona_registry.choose_persona(
            reg,
            goal="Fix a typo in docs",
            depth="narrow",
            kind="apply",
            effects="none",
        )
        self.assertEqual(pick.persona_id, "subagent.narrow_coder")


if __name__ == "__main__":
    unittest.main()
