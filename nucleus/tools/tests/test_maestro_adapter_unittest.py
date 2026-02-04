import unittest


class TestMaestroAdapter(unittest.TestCase):
    def test_build_command_default(self) -> None:
        from nucleus.tools import maestro_adapter

        class Args:
            mode = "test"
            flow = "flow.yaml"
            extra_args = []

        cmd = maestro_adapter.build_command("maestro", Args())
        self.assertEqual(cmd[:2], ["maestro", "test"])
        self.assertIn("flow.yaml", cmd)


if __name__ == "__main__":
    unittest.main()
