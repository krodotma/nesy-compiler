import unittest

import pathlib
import sys

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from topology_policy import choose_topology  # noqa: E402


class TestTopologyPolicy(unittest.TestCase):
    def test_sequential_default_single(self):
        d = choose_topology({"kind": "distill", "goal": "x"})
        self.assertEqual(d["topology"], "single")
        self.assertEqual(d["fanout"], 1)

    def test_parallelizable_low_tool_density_star(self):
        d = choose_topology({"kind": "distill", "parallelizable": True, "tool_density": 0.2, "coord_budget_tokens": 5000})
        self.assertEqual(d["topology"], "star")
        self.assertEqual(d["fanout"], 2)

    def test_tool_dense_forces_single(self):
        d = choose_topology({"kind": "distill", "parallelizable": True, "tool_density": 0.95, "coord_budget_tokens": 50000})
        self.assertEqual(d["topology"], "single")
        self.assertEqual(d["fanout"], 1)

    def test_low_budget_forces_single_even_if_hint(self):
        d = choose_topology({"kind": "distill", "parallelizable": True, "tool_density": 0.1, "coord_budget_tokens": 800, "topology_hint": "star"})
        self.assertEqual(d["topology"], "single")
        self.assertEqual(d["fanout"], 1)

    def test_hint_respected_when_safe(self):
        d = choose_topology({"kind": "distill", "parallelizable": True, "tool_density": 0.1, "coord_budget_tokens": 5000, "topology_hint": "peer_debate"})
        self.assertEqual(d["topology"], "peer_debate")
        self.assertEqual(d["fanout"], 2)


if __name__ == "__main__":
    unittest.main()
