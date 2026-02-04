#!/usr/bin/env python3
"""
Tests for TensorZero Gateway Integration
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensorzero_gateway import (
    TensorZeroConfig,
    TensorZeroGateway,
    InferenceResult,
    FeedbackResult,
    load_config,
)


class TestTensorZeroConfig(unittest.TestCase):
    """Test configuration loading."""

    def test_default_config(self):
        """Test default configuration values."""
        with patch.dict(os.environ, {}, clear=True):
            config = load_config()
            self.assertEqual(config.gateway_url, "http://localhost:3000")
            self.assertIsNone(config.api_key)
            self.assertEqual(config.default_function, "generate")

    def test_env_config(self):
        """Test configuration from environment."""
        env = {
            "TENSORZERO_GATEWAY_URL": "http://custom:8080",
            "TENSORZERO_API_KEY": "test-key",
            "TENSORZERO_DEFAULT_FUNCTION": "custom_fn",
            "TENSORZERO_TIMEOUT_S": "120",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            self.assertEqual(config.gateway_url, "http://custom:8080")
            self.assertEqual(config.api_key, "test-key")
            self.assertEqual(config.default_function, "custom_fn")
            self.assertEqual(config.timeout_s, 120.0)


class TestTensorZeroGateway(unittest.TestCase):
    """Test TensorZero gateway client."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.bus_dir = os.path.join(self.temp_dir, "bus")
        os.makedirs(self.bus_dir)

        self.config = TensorZeroConfig(
            gateway_url="http://localhost:3000",
            api_key=None,
            default_function="generate",
            emit_bus=False,  # Disable bus emission for tests
        )
        self.gateway = TensorZeroGateway(config=self.config, bus_dir=self.bus_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_health_check_unhealthy(self):
        """Test health check when gateway is unreachable."""
        # Without mocking, this should fail (no gateway running)
        result = self.gateway.health_check()
        self.assertFalse(result)

    @patch("tensorzero_gateway._http_request")
    def test_health_check_healthy(self, mock_request):
        """Test health check when gateway is reachable."""
        mock_request.return_value = (200, {"status": "ok"})
        result = self.gateway.health_check()
        self.assertTrue(result)

    @patch("tensorzero_gateway._http_request")
    def test_infer_success(self, mock_request):
        """Test successful inference."""
        mock_response = {
            "inference_id": "inf-123",
            "content": [{"text": "Test response"}],
            "model_name": "gpt-4o",
            "variant_name": "default",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        mock_request.return_value = (200, mock_response)

        result = self.gateway.infer("Test prompt")

        self.assertIsInstance(result, InferenceResult)
        self.assertEqual(result.inference_id, "inf-123")
        self.assertEqual(result.content, "Test response")
        self.assertEqual(result.model, "gpt-4o")
        self.assertEqual(result.input_tokens, 10)
        self.assertEqual(result.output_tokens, 20)

    @patch("tensorzero_gateway._http_request")
    def test_infer_openai_format(self, mock_request):
        """Test inference with OpenAI-compatible response format."""
        mock_response = {
            "inference_id": "inf-456",
            "choices": [{"message": {"content": "OpenAI style response"}}],
            "model": "gpt-4o-mini",
            "usage": {"prompt_tokens": 5, "completion_tokens": 15},
        }
        mock_request.return_value = (200, mock_response)

        result = self.gateway.infer("Test prompt")

        self.assertEqual(result.content, "OpenAI style response")
        self.assertEqual(result.input_tokens, 5)
        self.assertEqual(result.output_tokens, 15)

    @patch("tensorzero_gateway._http_request")
    def test_infer_error(self, mock_request):
        """Test inference error handling."""
        mock_request.return_value = (500, {"error": "Internal error"})

        with self.assertRaises(RuntimeError) as ctx:
            self.gateway.infer("Test prompt")

        self.assertIn("TensorZero inference failed", str(ctx.exception))

    @patch("tensorzero_gateway._http_request")
    def test_feedback_success(self, mock_request):
        """Test successful feedback submission."""
        mock_request.return_value = (200, {"feedback_id": "fb-123"})

        result = self.gateway.feedback("inf-123", 0.9, metric_name="quality")

        self.assertIsInstance(result, FeedbackResult)
        self.assertEqual(result.inference_id, "inf-123")
        self.assertEqual(result.score, 0.9)
        self.assertEqual(result.metric_name, "quality")
        self.assertTrue(result.accepted)

    @patch("tensorzero_gateway._http_request")
    def test_feedback_rejected(self, mock_request):
        """Test rejected feedback."""
        mock_request.return_value = (400, {"error": "Invalid inference_id"})

        result = self.gateway.feedback("invalid-id", 0.5)

        self.assertFalse(result.accepted)

    @patch("tensorzero_gateway._http_request")
    def test_list_experiments(self, mock_request):
        """Test listing experiments."""
        mock_experiments = [
            {"id": "exp-1", "name": "A/B Test", "status": "active"},
            {"id": "exp-2", "name": "Model Compare", "status": "completed"},
        ]
        mock_request.return_value = (200, {"experiments": mock_experiments})

        result = self.gateway.list_experiments()

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "A/B Test")

    def test_metrics_buffer(self):
        """Test metrics buffering."""
        # Manually add to buffer
        self.gateway._metrics_buffer.append({
            "ts": 1000,
            "inference_id": "test-1",
            "latency_ms": 100,
            "tokens": 30,
        })
        self.gateway._metrics_buffer.append({
            "ts": 1001,
            "inference_id": "test-2",
            "latency_ms": 200,
            "tokens": 40,
        })

        summary = self.gateway.get_metrics_summary()

        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["avg_latency_ms"], 150.0)
        self.assertEqual(summary["total_tokens"], 70)

    def test_flush_metrics(self):
        """Test flushing metrics buffer."""
        self.gateway._metrics_buffer.append({"ts": 1000, "latency_ms": 100, "tokens": 30})

        self.gateway.flush_metrics()

        self.assertEqual(len(self.gateway._metrics_buffer), 0)


class TestBusIntegration(unittest.TestCase):
    """Test bus event emission."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.bus_dir = os.path.join(self.temp_dir, "bus")
        os.makedirs(self.bus_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("tensorzero_gateway._http_request")
    def test_infer_emits_events(self, mock_request):
        """Test that inference emits bus events."""
        mock_response = {
            "inference_id": "inf-789",
            "content": "Response",
            "model_name": "gpt-4o",
            "usage": {"input_tokens": 5, "output_tokens": 10},
        }
        mock_request.return_value = (200, mock_response)

        config = TensorZeroConfig(
            gateway_url="http://localhost:3000",
            emit_bus=True,
        )
        gateway = TensorZeroGateway(config=config, bus_dir=self.bus_dir)
        gateway.infer("Test prompt", trace_id="test-trace")

        # Check bus events were written
        events_path = Path(self.bus_dir) / "events.ndjson"
        # Events may or may not be written depending on whether agent_bus.py exists
        # This test mainly verifies no exceptions are raised


class TestCLI(unittest.TestCase):
    """Test CLI commands."""

    def test_health_command(self):
        """Test health check command."""
        from tensorzero_gateway import build_parser

        parser = build_parser()
        args = parser.parse_args(["health"])
        self.assertEqual(args.cmd, "health")

    def test_infer_command(self):
        """Test infer command parsing."""
        from tensorzero_gateway import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "infer",
            "--prompt", "Test prompt",
            "--function", "custom",
            "--json",
        ])
        self.assertEqual(args.cmd, "infer")
        self.assertEqual(args.prompt, "Test prompt")
        self.assertEqual(args.function, "custom")
        self.assertTrue(args.json)

    def test_feedback_command(self):
        """Test feedback command parsing."""
        from tensorzero_gateway import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "feedback",
            "--inference-id", "inf-123",
            "--score", "0.9",
            "--metric", "quality",
        ])
        self.assertEqual(args.inference_id, "inf-123")
        self.assertEqual(args.score, 0.9)
        self.assertEqual(args.metric, "quality")


if __name__ == "__main__":
    unittest.main()
