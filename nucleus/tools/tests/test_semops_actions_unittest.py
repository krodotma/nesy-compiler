import unittest


class TestSemopsActions(unittest.TestCase):
    def test_builtin_op_actions_include_clone(self):
        from nucleus.tools.semops_actions import derive_ui_actions, infer_flow_hints

        op = {
            "user_defined": False,
            "tool": "nucleus/tools/realagents_operator.py",
            "bus_topic": "rd.tasks.dispatch",
            "ui": {"route": "/semops", "component": "SemopsEditor"},
            "domain": "coordination",
            "category": "dispatch",
            "name": "REALAGENTS",
            "description": "Formal assignment operator",
        }
        actions = derive_ui_actions(operator_key="REALAGENTS", op=op)
        ids = {a.get("id") for a in actions}
        self.assertIn("clone", ids)
        self.assertIn("open_tool", ids)
        self.assertIn("open_bus", ids)
        self.assertIn("open_ui", ids)
        self.assertIn("invoke", ids)

        hints = infer_flow_hints(operator_key="REALAGENTS", op=op)
        self.assertIn("tool", hints)
        self.assertIn("bus", hints)
        self.assertIn("ui", hints)

    def test_user_op_actions_include_edit_and_delete(self):
        from nucleus.tools.semops_actions import derive_ui_actions

        op = {"user_defined": True, "domain": "user", "category": "custom", "name": "MYOP"}
        actions = derive_ui_actions(operator_key="MYOP", op=op)
        ids = {a.get("id") for a in actions}
        self.assertIn("edit", ids)
        self.assertIn("delete", ids)
        self.assertNotIn("clone", ids)


if __name__ == "__main__":
    unittest.main()

