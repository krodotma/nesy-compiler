#!/usr/bin/env python3
"""
Tests for WasmEdgeRuntime - WASM-based sandboxed execution.

Tests cover:
- Runtime detection
- WASM execution (dry-run)
- WAT compilation
- wasi_nn inference
- Network modes
- File mappings
- AgentSandbox integration
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from wasmedge_runtime import (
    WasmEdgeRuntime,
    WasmEdgeDetector,
    WasmConfig,
    WasmResult,
    InferenceResult,
    FileMapping,
    NetworkMode,
    WasiPlugin,
    AgentSandbox,
)


class TestWasmEdgeDetector:
    """Tests for WasmEdgeDetector."""

    def test_is_installed_missing(self):
        """Should return False when wasmedge not installed."""
        with patch("shutil.which", return_value=None):
            assert WasmEdgeDetector.is_installed() is False

    def test_is_installed_present(self):
        """Should return True when wasmedge is installed."""
        with patch("shutil.which", return_value="/usr/local/bin/wasmedge"):
            assert WasmEdgeDetector.is_installed() is True

    def test_get_version_not_installed(self):
        """Should return None when not installed."""
        with patch.object(WasmEdgeDetector, "is_installed", return_value=False):
            assert WasmEdgeDetector.get_version() is None

    def test_get_capabilities(self):
        """Capabilities should include all fields."""
        caps = WasmEdgeDetector.get_capabilities()
        assert "installed" in caps
        assert "version" in caps
        assert "plugins" in caps
        assert "wasi_nn" in caps
        assert "ggml_backend" in caps


class TestWasmConfig:
    """Tests for WasmConfig."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = WasmConfig()
        assert config.max_memory_pages == 256
        assert config.timeout_ms == 30000
        assert config.enable_simd is True
        assert config.enable_threads is False  # Disabled for safety

    def test_custom_config(self):
        """Custom config should override defaults."""
        config = WasmConfig(
            max_memory_pages=512,
            timeout_ms=60000,
            enable_threads=True,
        )
        assert config.max_memory_pages == 512
        assert config.timeout_ms == 60000
        assert config.enable_threads is True


class TestFileMapping:
    """Tests for FileMapping."""

    def test_readonly_default(self):
        """File mappings should be readonly by default."""
        mapping = FileMapping(
            host_path="/host/path",
            wasm_path="/wasm/path",
        )
        assert mapping.readonly is True

    def test_writable_mapping(self):
        """Writable mappings should be configurable."""
        mapping = FileMapping(
            host_path="/host/path",
            wasm_path="/wasm/path",
            readonly=False,
        )
        assert mapping.readonly is False


class TestWasmResult:
    """Tests for WasmResult."""

    def test_result_fields(self):
        """Result should have all expected fields."""
        result = WasmResult(
            exit_code=0,
            stdout="output",
            stderr="",
            duration_ms=100,
            memory_used_bytes=1024,
            runtime_version="0.13.0",
            plugins_loaded=["wasi_nn"],
            trace_id="test-123",
        )

        assert result.exit_code == 0
        assert result.stdout == "output"
        assert result.duration_ms == 100
        assert result.memory_used_bytes == 1024
        assert "wasi_nn" in result.plugins_loaded
        assert result.trace_id == "test-123"


class TestInferenceResult:
    """Tests for InferenceResult."""

    def test_inference_result_fields(self):
        """Inference result should have all expected fields."""
        result = InferenceResult(
            output="Hello, world!",
            tokens_generated=3,
            duration_ms=500,
            tokens_per_second=6.0,
            model_name="llama-7b.gguf",
            backend="ggml",
            trace_id="infer-123",
        )

        assert result.output == "Hello, world!"
        assert result.tokens_generated == 3
        assert result.duration_ms == 500
        assert result.tokens_per_second == 6.0
        assert result.model_name == "llama-7b.gguf"
        assert result.backend == "ggml"


class TestWasmEdgeRuntime:
    """Tests for WasmEdgeRuntime."""

    def test_init_dry_run(self):
        """Runtime should init in dry-run mode when wasmedge missing."""
        with patch.object(WasmEdgeDetector, "is_installed", return_value=False):
            runtime = WasmEdgeRuntime()
            assert runtime.dry_run is True

    def test_init_with_wasmedge(self):
        """Runtime should not be in dry-run when wasmedge installed."""
        with patch.object(WasmEdgeDetector, "is_installed", return_value=True):
            runtime = WasmEdgeRuntime()
            assert runtime.dry_run is False

    def test_execute_file_not_found(self):
        """Execute should fail gracefully for missing WASM file."""
        runtime = WasmEdgeRuntime()
        result = runtime.execute(
            wasm_path="/nonexistent/module.wasm",
            trace_id="test-404",
        )

        assert result.exit_code == 1
        assert "not found" in result.stderr.lower()

    def test_execute_dry_run(self):
        """Dry-run execution should return mock result."""
        with patch.object(WasmEdgeDetector, "is_installed", return_value=False):
            runtime = WasmEdgeRuntime()

            # Create temp WASM file
            with tempfile.NamedTemporaryFile(suffix=".wasm", delete=False) as f:
                f.write(b"\x00asm")  # Minimal WASM header
                wasm_path = f.name

            try:
                result = runtime.execute(
                    wasm_path=wasm_path,
                    args=["--test"],
                    trace_id="dry-run-test",
                )

                assert result.exit_code == 0
                assert "dry_run" in result.stdout

                data = json.loads(result.stdout)
                assert data["status"] == "dry_run"
                assert data["wasm_path"] == wasm_path
            finally:
                Path(wasm_path).unlink()

    def test_build_base_args(self):
        """Base args should include memory limits and features."""
        runtime = WasmEdgeRuntime()
        args = runtime._build_base_args()

        assert "wasmedge" in args
        assert "--memory-page-limit" in args
        assert "--gas-limit" in args
        assert "--disable-simd" not in args

        disabled_simd = WasmEdgeRuntime(config=WasmConfig(enable_simd=False))._build_base_args()
        assert "--disable-simd" in disabled_simd

    def test_add_env_vars(self):
        """Environment variables should be added to command."""
        runtime = WasmEdgeRuntime()
        args = ["wasmedge"]
        env = {"MY_VAR": "value", "OTHER": "test"}

        args = runtime._add_env_vars(args, env)

        assert "--env" in args
        assert any("MY_VAR=value" in a for a in args)
        assert any("OTHER=test" in a for a in args)

    def test_add_file_mappings(self):
        """File mappings should be added to command."""
        runtime = WasmEdgeRuntime()
        args = ["wasmedge"]
        mappings = [
            FileMapping("/host/data", "/wasm/data", readonly=True),
            FileMapping("/host/output", "/wasm/output", readonly=False),
        ]

        args = runtime._add_file_mappings(args, mappings)

        assert "--dir" in args

    def test_infer_no_wasi_nn(self):
        """Inference should fail gracefully without wasi_nn."""
        with patch.object(WasmEdgeDetector, "has_wasi_nn", return_value=False):
            runtime = WasmEdgeRuntime()
            result = runtime.infer(
                model_path="/models/test.gguf",
                prompt="Hello",
            )

            # Should return empty result (no wasi_nn)
            assert result.tokens_generated == 0

    def test_get_capabilities(self):
        """Capabilities should include runtime config."""
        runtime = WasmEdgeRuntime()
        caps = runtime.get_capabilities()

        assert "dry_run" in caps
        assert "config" in caps
        assert "max_memory_pages" in caps["config"]
        assert "timeout_ms" in caps["config"]


class TestNetworkMode:
    """Tests for NetworkMode enum."""

    def test_network_mode_values(self):
        """Network modes should have correct values."""
        assert NetworkMode.DISABLED.value == "disabled"
        assert NetworkMode.LOCALHOST_ONLY.value == "localhost"
        assert NetworkMode.ALLOWLIST.value == "allowlist"
        assert NetworkMode.FULL.value == "full"


class TestWasiPlugin:
    """Tests for WasiPlugin enum."""

    def test_plugin_values(self):
        """Plugin names should be correct."""
        assert WasiPlugin.WASI_NN.value == "wasi_nn"
        assert WasiPlugin.WASI_GGML.value == "wasi_nn-ggml"


class TestAgentSandbox:
    """Tests for AgentSandbox."""

    def test_init(self):
        """AgentSandbox should initialize with WasmEdgeRuntime."""
        sandbox = AgentSandbox()
        assert sandbox.wasm_runtime is not None

    def test_execute_tool_wasm_mode(self):
        """Tool execution in WASM mode should use WasmEdgeRuntime."""
        sandbox = AgentSandbox()

        result = sandbox.execute_tool(
            tool_name="hash",
            tool_input={"data": "test"},
            execution_mode="wasm",
            trace_id="tool-test",
        )

        # Should return result dict
        assert "success" in result
        assert "trace_id" in result
        assert result["trace_id"] == "tool-test"

    def test_execute_tool_auto_selection(self):
        """Auto mode should select appropriate runtime."""
        sandbox = AgentSandbox()

        # "hash" should be auto-selected as WASM
        result = sandbox.execute_tool(
            tool_name="hash",
            tool_input={"data": "test"},
            execution_mode="auto",
        )

        assert "success" in result

    def test_get_capabilities(self):
        """Capabilities should include both runtimes."""
        sandbox = AgentSandbox()
        caps = sandbox.get_capabilities()

        assert "wasm" in caps
        assert "containers" in caps


class TestWATExecution:
    """Tests for WAT (WebAssembly Text) execution."""

    def test_execute_wat_no_wat2wasm(self):
        """WAT execution should fail without wat2wasm."""
        with patch("shutil.which", return_value=None):
            runtime = WasmEdgeRuntime()
            result = runtime.execute_wat(
                wat_code="(module)",
                trace_id="wat-test",
            )

            assert result.exit_code == 1
            assert "wat2wasm" in result.stderr


class TestAOTCompilation:
    """Tests for AOT compilation."""

    def test_compile_aot_file_not_found(self):
        """AOT compilation should fail for missing file."""
        runtime = WasmEdgeRuntime()
        success, result = runtime.compile_to_aot("/nonexistent.wasm")

        assert success is False
        assert "not found" in result.lower()


class TestIntegration:
    """Integration tests (require WasmEdge installation)."""

    @pytest.mark.skipif(
        not WasmEdgeDetector.is_installed(),
        reason="WasmEdge not installed"
    )
    def test_real_wasm_execution(self):
        """Test actual WASM execution if WasmEdge available."""
        runtime = WasmEdgeRuntime()

        # Create minimal valid WASM module
        # This is a module that exports nothing but is valid
        wat_code = """
        (module
            (func (export "_start"))
        )
        """

        result = runtime.execute_wat(
            wat_code=wat_code,
            trace_id="integration-test",
        )

        # If wat2wasm and wasmedge both available, should succeed
        if result.exit_code != 1 or "wat2wasm" not in result.stderr:
            assert result.exit_code == 0

    @pytest.mark.skipif(
        not WasmEdgeDetector.is_installed() or not WasmEdgeDetector.has_wasi_nn(),
        reason="WasmEdge with wasi_nn not available"
    )
    def test_wasi_nn_inference(self):
        """Test wasi_nn inference if available."""
        runtime = WasmEdgeRuntime()

        # This would require an actual model and inference WASM
        # Skip for now as it requires significant setup
        pytest.skip("Requires model and inference WASM setup")


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_execute_empty_wasm_path(self):
        """Empty WASM path should return error before dry-run check."""
        runtime = WasmEdgeRuntime()
        result = runtime.execute(wasm_path="")

        # Empty path should fail validation regardless of dry-run mode
        assert result.exit_code == 1
        assert "not found" in result.stderr.lower()

    def test_execute_with_unicode_env(self):
        """Unicode environment variables should be handled."""
        with patch.object(WasmEdgeDetector, "is_installed", return_value=False):
            runtime = WasmEdgeRuntime()

            with tempfile.NamedTemporaryFile(suffix=".wasm", delete=False) as f:
                f.write(b"\x00asm")
                wasm_path = f.name

            try:
                result = runtime.execute(
                    wasm_path=wasm_path,
                    env={"UNICODE_VAR": "test"},
                )

                # Should handle without error
                assert result.exit_code == 0
            finally:
                Path(wasm_path).unlink()

    def test_config_with_zero_timeout(self):
        """Zero timeout should be handled."""
        config = WasmConfig(timeout_ms=0)
        runtime = WasmEdgeRuntime(config=config)

        # Should not crash
        assert runtime.config.timeout_ms == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
