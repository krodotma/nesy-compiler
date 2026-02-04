import json
import pathlib
import sys
import unittest

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
ROOT = TOOLS_DIR.parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import cagent_registry  # noqa: E402


class TestCagentRegistry(unittest.TestCase):
    def test_registry_spec_validates(self):
        path = ROOT / "nucleus" / "specs" / "cagent_registry.json"
        obj = json.loads(path.read_text(encoding="utf-8"))
        ok, errs = cagent_registry.validate_registry(obj)
        self.assertTrue(ok, errs)

    def test_resolve_actor_registry_hit(self):
        registry = {
            "defaults": {
                "citizen_class": "superworker",
                "citizen_tier": "limited",
                "bootstrap_profile": "minimal",
                "scope_allowlist": [],
            },
            "class_aliases": {"sagent": "superagent"},
            "tier_aliases": {"full": "full", "limited": "limited"},
            "actors": [
                {"actor": "codex", "citizen_class": "superagent", "citizen_tier": "full"}
            ],
        }
        profile = cagent_registry.resolve_actor("codex", registry, overrides=None, allow_override=False)
        self.assertEqual(profile.citizen_class, "superagent")
        self.assertEqual(profile.citizen_tier, "full")
        self.assertEqual(profile.source, "registry")

    def test_resolve_actor_override(self):
        registry = {
            "defaults": {"citizen_class": "superworker", "citizen_tier": "limited"},
            "class_aliases": {"swagent": "superworker"},
            "tier_aliases": {"limited": "limited"},
            "actors": [{"actor": "codex", "citizen_class": "superagent"}],
        }
        profile = cagent_registry.resolve_actor(
            "codex",
            registry,
            overrides={"citizen_class": "swagent"},
            allow_override=True,
        )
        self.assertEqual(profile.citizen_class, "superworker")


if __name__ == "__main__":
    unittest.main()
