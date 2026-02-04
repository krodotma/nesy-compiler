import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


class TestAgentHomeInit(unittest.TestCase):
    def test_select_agent_home_base_primary(self) -> None:
        from nucleus.tools.agent_home_init import select_agent_home_base

        with TemporaryDirectory() as td:
            root = Path(td)
            base = select_agent_home_base(root, bus_dir=None, actor="tester")
            self.assertEqual(base, root / ".pluribus" / "agent_homes")
            self.assertTrue(base.exists())

    def test_select_agent_home_base_fallback_on_permission(self) -> None:
        from nucleus.tools import agent_home_init

        with TemporaryDirectory() as td:
            root = Path(td)
            primary = root / ".pluribus" / "agent_homes"
            fallback = root / ".pluribus_local" / "agent_homes"

            def fake_ensure_dir(path: Path) -> None:
                if path == primary:
                    raise PermissionError("no access")
                path.mkdir(parents=True, exist_ok=True)

            with patch.object(agent_home_init, "ensure_dir", side_effect=fake_ensure_dir):
                base = agent_home_init.select_agent_home_base(root, bus_dir=None, actor="tester")

            self.assertEqual(base, fallback)
            self.assertTrue(fallback.exists())


if __name__ == "__main__":
    unittest.main()
