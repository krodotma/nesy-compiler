import unittest
from unittest.mock import patch


class TestAgentSAdapter(unittest.TestCase):
    def test_missing_grounding_raises(self) -> None:
        from nucleus.tools import agent_s_adapter

        class Args:
            provider = None
            model = None
            model_url = None
            model_api_key = None
            model_temperature = None
            ground_provider = None
            ground_url = None
            ground_api_key = None
            ground_model = None
            grounding_width = None
            grounding_height = None
            max_trajectory_length = None
            enable_reflection = None
            enable_local_env = None

        with self.assertRaises(ValueError):
            agent_s_adapter.build_agent_s_args(Args())

    def test_entrypoint_override(self) -> None:
        from nucleus.tools import agent_s_adapter

        with patch.dict("os.environ", {"AGENT_S_ENTRYPOINT": "agent_s"}, clear=False):
            entry = agent_s_adapter.resolve_entrypoint()
            self.assertEqual(entry, ["agent_s"])

    def test_args_include_required_fields(self) -> None:
        from nucleus.tools import agent_s_adapter

        class Args:
            provider = "openai"
            model = "gpt-5-2025-08-07"
            model_url = ""
            model_api_key = ""
            model_temperature = None
            ground_provider = "huggingface"
            ground_url = "http://localhost:8080"
            ground_api_key = ""
            ground_model = "ui-tars-1.5-7b"
            grounding_width = 1920
            grounding_height = 1080
            max_trajectory_length = None
            enable_reflection = True
            enable_local_env = False

        args_list = agent_s_adapter.build_agent_s_args(Args())
        self.assertIn("--ground_provider", args_list)
        self.assertIn("--ground_model", args_list)
        self.assertIn("--grounding_width", args_list)


if __name__ == "__main__":
    unittest.main()
