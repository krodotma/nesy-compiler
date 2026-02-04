import pathlib
import sys
import unittest


TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import vps_session  # noqa: E402


class TestVPSSessionWebOnlyDefaults(unittest.TestCase):
    def test_default_fallback_order_is_web_only(self):
        sess = vps_session.VPSSession()
        self.assertEqual(sess.fallback_order[:3], ["chatgpt-web", "claude-web", "gemini-web"])
        self.assertNotIn("mock", sess.fallback_order)
        self.assertNotIn("codex-cli", sess.fallback_order)
        self.assertNotIn("claude-cli", sess.fallback_order)
        self.assertNotIn("gemini-cli", sess.fallback_order)

    def test_default_providers_include_web_only(self):
        sess = vps_session.VPSSession()
        keys = set(sess.providers.keys())
        self.assertTrue({"chatgpt-web", "claude-web", "gemini-web"}.issubset(keys))
        # Mock is internal-only and should not be present unless explicitly enabled.
        self.assertNotIn("mock", keys)


if __name__ == "__main__":
    unittest.main()

