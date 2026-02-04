import pathlib
import sys
import tempfile
import unittest
from unittest import mock
import os

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import plurichat  # noqa: E402


class TestPluriChatStrpTrigger(unittest.TestCase):
    def test_one_shot_uses_strp_when_topology_star(self):
        # Choose a prompt that should classify as deep + distill (and effects=none),
        # which yields star fanout in lens_collimator/topology_policy.
        prompt = "Distill the architecture into a short spec with explicit tests and gates."

        class FakeStatus:
            def __init__(self, available: bool):
                self.available = available

        def fake_status():
            return {"mock": FakeStatus(True)}

        with tempfile.TemporaryDirectory() as td, \
             mock.patch.dict(os.environ, {"PLURIBUS_ALLOW_MOCK": "1"}, clear=False), \
             mock.patch.object(plurichat, "get_all_provider_status", side_effect=fake_status), \
             mock.patch.object(plurichat, "dispatch_to_strp_queue", return_value=("r1", True)) as m_dispatch, \
             mock.patch.object(plurichat, "run_strp_worker_once", return_value=True) as m_worker, \
             mock.patch.object(plurichat, "wait_for_strp_response", return_value="ok") as m_wait, \
             mock.patch.object(plurichat, "execute_chat_direct") as m_direct, \
             mock.patch("builtins.print"):
            m_direct.side_effect = AssertionError("direct path should not be used when topology=star")

            bus_dir = pathlib.Path(td) / "bus"
            code = plurichat.main(["--ask", prompt, "--provider", "mock", "--mode", "direct", "--bus-dir", str(bus_dir)])
            self.assertEqual(code, 0)
            self.assertTrue(m_dispatch.called)
            self.assertTrue(m_worker.called)
            self.assertTrue(m_wait.called)


if __name__ == "__main__":
    unittest.main()
