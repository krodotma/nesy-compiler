import unittest

from qa_action_executor import build_allowed_actions, evaluate_action, action_key


class TestQAActionExecutor(unittest.TestCase):
    def test_action_key(self):
        self.assertEqual(action_key("plan", "restart.dashboard"), "plan:restart.dashboard")

    def test_unknown_action_is_rejected(self):
        allowed = build_allowed_actions([])
        status, command = evaluate_action({"id": "unknown.action"}, allowed, allow_root=True)
        self.assertEqual(status, "not_allowed")
        self.assertIsNone(command)

    def test_manual_action_skips(self):
        allowed = build_allowed_actions([])
        status, command = evaluate_action({"id": "telemetry.reduce"}, allowed, allow_root=True)
        self.assertEqual(status, "manual")
        self.assertIsNone(command)

    def test_command_mismatch_skips(self):
        allowed = build_allowed_actions([])
        status, command = evaluate_action(
            {"id": "restart.dashboard", "command": "systemctl restart not-dashboard"},
            allowed,
            allow_root=True,
        )
        self.assertEqual(status, "command_mismatch")
        self.assertIsNone(command)

    def test_requires_root(self):
        allowed = build_allowed_actions([])
        status, command = evaluate_action({"id": "restart.dashboard"}, allowed, allow_root=False)
        self.assertEqual(status, "root_not_allowed")
        self.assertIsNone(command)

    def test_allowed_action_ok(self):
        allowed = build_allowed_actions([])
        status, command = evaluate_action({"id": "restart.dashboard"}, allowed, allow_root=True)
        self.assertEqual(status, "ok")
        self.assertEqual(command, "systemctl restart pluribus-dashboard")


if __name__ == "__main__":
    unittest.main()
