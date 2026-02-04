#!/usr/bin/env python3
"""
WasmEdge Runtime: WASM-based Sandboxed Execution
=================================================

Implements WASM sandboxing for Pluribus agent tool execution using WasmEdge.

Features:
- wasi_nn: Neural network inference in WASM
- wasi_socket: Controlled network access
- wasi_filesystem: Sandboxed filesystem access
- Agent sandbox integration for tool calls

Capabilities:
- Execute pre-compiled WASM modules
- Run WASI-compatible programs
- Neural inference via wasi_nn plugin
- HTTP/socket operations via wasi_socket

Usage:
    runtime = WasmEdgeRuntime()
    result = runtime.execute(
        wasm_path="/path/to/module.wasm",
        args=["--input", "data.json"],
        trace_id="abc-123"
    )

    # Neural inference
    result = runtime.infer(
        model_path="/models/llama.gguf",
        prompt="Hello, world!",
        backend="ggml"
    )
"""
from __future__ import annotations

import os
import sys
import json
import subprocess
import uuid
import time
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum


class WasiPlugin(Enum):
    """Available WASI plugins."""
    WASI_NN = "wasi_nn"
    WASI_CRYPTO = "wasi_crypto"
    WASI_SOCKET = "wasmedge_rustls"
    WASI_TENSORFLOW = "wasi_nn-tensorflowlite"
    WASI_GGML = "wasi_nn-ggml"


class NetworkMode(Enum):
    """Network access modes for WASM execution."""
    DISABLED = "disabled"
    LOCALHOST_ONLY = "localhost"
    ALLOWLIST = "allowlist"
    FULL = "full"  # Dangerous, requires explicit opt-in


@dataclass
class WasmConfig:
    """Configuration for WASM execution."""
    max_memory_pages: int = 256  # 16MB (64KB per page)
    max_table_elements: int = 1024
    timeout_ms: int = 30000
    enable_bulk_memory: bool = True
    enable_reference_types: bool = True
    enable_simd: bool = True
    enable_threads: bool = False  # Disabled by default for safety
    stack_size: int = 1048576  # 1MB


@dataclass
class FileMapping:
    """Host to WASM filesystem mapping."""
    host_path: str
    wasm_path: str
    readonly: bool = True


@dataclass
class WasmResult:
    """Result from WASM execution."""
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    memory_used_bytes: int
    runtime_version: str
    plugins_loaded: List[str]
    trace_id: Optional[str] = None


@dataclass
class InferenceResult:
    """Result from neural inference."""
    output: str
    tokens_generated: int
    duration_ms: int
    tokens_per_second: float
    model_name: str
    backend: str
    trace_id: Optional[str] = None


class WasmEdgeDetector:
    """Detects WasmEdge installation and capabilities."""

    @staticmethod
    def is_installed() -> bool:
        """Check if WasmEdge is installed."""
        return shutil.which("wasmedge") is not None

    @staticmethod
    def get_version() -> Optional[str]:
        """Get WasmEdge version."""
        if not WasmEdgeDetector.is_installed():
            return None
        try:
            result = subprocess.run(
                ["wasmedge", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    @staticmethod
    def list_plugins() -> List[str]:
        """List available WasmEdge plugins."""
        plugins = []
        plugin_dir = Path("/usr/local/lib/wasmedge")

        # Check common plugin locations
        for search_path in [
            plugin_dir,
            Path.home() / ".wasmedge" / "plugin",
            Path("/opt/wasmedge/plugin"),
        ]:
            if search_path.exists():
                for plugin_file in search_path.glob("libwasmedge*.so"):
                    plugin_name = plugin_file.stem.replace("libwasmedge", "")
                    if plugin_name:
                        plugins.append(plugin_name)

        return list(set(plugins))

    @staticmethod
    def has_wasi_nn() -> bool:
        """Check if wasi_nn plugin is available."""
        plugins = WasmEdgeDetector.list_plugins()
        return any("wasi_nn" in p for p in plugins)

    @staticmethod
    def has_ggml_backend() -> bool:
        """Check if GGML backend for wasi_nn is available."""
        plugins = WasmEdgeDetector.list_plugins()
        return any("ggml" in p.lower() for p in plugins)

    @classmethod
    def get_capabilities(cls) -> Dict[str, Any]:
        """Return full capabilities report."""
        return {
            "installed": cls.is_installed(),
            "version": cls.get_version(),
            "plugins": cls.list_plugins(),
            "wasi_nn": cls.has_wasi_nn(),
            "ggml_backend": cls.has_ggml_backend(),
        }


class WasmEdgeRuntime:
    """WasmEdge runtime wrapper for sandboxed execution."""

    def __init__(
        self,
        config: Optional[WasmConfig] = None,
        plugin_dir: Optional[str] = None,
    ):
        self.config = config or WasmConfig()
        self.plugin_dir = plugin_dir or os.environ.get(
            "WASMEDGE_PLUGIN_PATH",
            "/usr/local/lib/wasmedge"
        )
        self._validate_installation()

    def _validate_installation(self) -> None:
        """Validate WasmEdge installation."""
        if not WasmEdgeDetector.is_installed():
            # Set dry-run mode
            self.dry_run = True
        else:
            self.dry_run = False

    def _build_base_args(self) -> List[str]:
        """Build base WasmEdge arguments."""
        args = ["wasmedge"]

        # Memory limits
        args.extend(["--max-memory-pages", str(self.config.max_memory_pages)])

        # Features
        if self.config.enable_bulk_memory:
            args.append("--enable-bulk-memory")
        if self.config.enable_reference_types:
            args.append("--enable-reference-types")
        if self.config.enable_simd:
            args.append("--enable-simd")
        if self.config.enable_threads:
            args.append("--enable-threads")

        # Timeout (in gas units, approximate)
        gas_limit = self.config.timeout_ms * 1000000  # Rough approximation
        args.extend(["--gas-limit", str(gas_limit)])

        return args

    def _add_file_mappings(
        self,
        args: List[str],
        mappings: List[FileMapping]
    ) -> List[str]:
        """Add WASI file mappings to command."""
        for mapping in mappings:
            mode = "readonly" if mapping.readonly else ""
            # WasmEdge uses --dir for WASI filesystem access
            args.extend([
                "--dir", f"{mapping.wasm_path}:{mapping.host_path}"
            ])
        return args

    def _add_env_vars(
        self,
        args: List[str],
        env: Dict[str, str]
    ) -> List[str]:
        """Add environment variables to WASM execution."""
        for key, value in env.items():
            args.extend(["--env", f"{key}={value}"])
        return args

    def execute(
        self,
        wasm_path: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        file_mappings: Optional[List[FileMapping]] = None,
        stdin_data: Optional[str] = None,
        trace_id: Optional[str] = None,
        network_mode: NetworkMode = NetworkMode.DISABLED,
    ) -> WasmResult:
        """
        Execute a WASM module.

        Args:
            wasm_path: Path to .wasm file
            args: Command-line arguments for the WASM program
            env: Environment variables
            file_mappings: Host to WASM filesystem mappings
            stdin_data: Data to pass via stdin
            trace_id: Trace ID for observability
            network_mode: Network access mode

        Returns:
            WasmResult with execution details
        """
        start_time = time.monotonic()
        trace_id = trace_id or str(uuid.uuid4())

        # Validate WASM file exists
        wasm_file = Path(wasm_path) if wasm_path else None
        if not wasm_path or not wasm_file or not wasm_file.exists() or not wasm_file.is_file():
            return WasmResult(
                exit_code=1,
                stdout="",
                stderr=f"WASM file not found: {wasm_path}",
                duration_ms=0,
                memory_used_bytes=0,
                runtime_version="unknown",
                plugins_loaded=[],
                trace_id=trace_id,
            )

        # Handle dry-run mode
        if self.dry_run:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            return WasmResult(
                exit_code=0,
                stdout=json.dumps({
                    "status": "dry_run",
                    "wasm_path": wasm_path,
                    "args": args or [],
                    "network_mode": network_mode.value,
                    "capabilities": WasmEdgeDetector.get_capabilities(),
                }),
                stderr="(dry run mode - WasmEdge not installed)",
                duration_ms=duration_ms,
                memory_used_bytes=0,
                runtime_version="mock",
                plugins_loaded=[],
                trace_id=trace_id,
            )

        # Build command
        cmd = self._build_base_args()

        # Add environment variables
        env_vars = (env or {}).copy()
        env_vars["PLURIBUS_TRACE_ID"] = trace_id
        env_vars["PLURIBUS_ISOLATION"] = "wasm"
        cmd = self._add_env_vars(cmd, env_vars)

        # Add file mappings
        if file_mappings:
            cmd = self._add_file_mappings(cmd, file_mappings)

        # Network mode (requires wasi_socket plugin)
        if network_mode == NetworkMode.LOCALHOST_ONLY:
            cmd.extend(["--env", "WASI_SOCKET_ALLOWLIST=127.0.0.1"])
        elif network_mode == NetworkMode.ALLOWLIST:
            # Would need allowlist specification
            pass
        # DISABLED is the default (no socket plugin loaded)

        # Add WASM file and arguments
        cmd.append(str(wasm_file))
        if args:
            cmd.extend(args)

        # Execute
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_ms / 1000,
                input=stdin_data,
            )
            duration_ms = int((time.monotonic() - start_time) * 1000)

            return WasmResult(
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                duration_ms=duration_ms,
                memory_used_bytes=0,  # Would need instrumentation
                runtime_version=WasmEdgeDetector.get_version() or "unknown",
                plugins_loaded=WasmEdgeDetector.list_plugins(),
                trace_id=trace_id,
            )
        except subprocess.TimeoutExpired:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            return WasmResult(
                exit_code=124,
                stdout="",
                stderr=f"Timeout after {self.config.timeout_ms}ms",
                duration_ms=duration_ms,
                memory_used_bytes=0,
                runtime_version=WasmEdgeDetector.get_version() or "unknown",
                plugins_loaded=[],
                trace_id=trace_id,
            )
        except Exception as e:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            return WasmResult(
                exit_code=1,
                stdout="",
                stderr=str(e),
                duration_ms=duration_ms,
                memory_used_bytes=0,
                runtime_version=WasmEdgeDetector.get_version() or "unknown",
                plugins_loaded=[],
                trace_id=trace_id,
            )

    def execute_wat(
        self,
        wat_code: str,
        args: Optional[List[str]] = None,
        trace_id: Optional[str] = None,
    ) -> WasmResult:
        """
        Execute WebAssembly Text (WAT) code.

        Compiles WAT to WASM on-the-fly and executes.
        """
        trace_id = trace_id or str(uuid.uuid4())

        # Check for wat2wasm (from wabt)
        if not shutil.which("wat2wasm"):
            return WasmResult(
                exit_code=1,
                stdout="",
                stderr="wat2wasm not found (install wabt)",
                duration_ms=0,
                memory_used_bytes=0,
                runtime_version="unknown",
                plugins_loaded=[],
                trace_id=trace_id,
            )

        # Create temp files
        with tempfile.NamedTemporaryFile(
            suffix=".wat", mode="w", delete=False
        ) as wat_file:
            wat_file.write(wat_code)
            wat_path = wat_file.name

        wasm_path = wat_path.replace(".wat", ".wasm")

        try:
            # Compile WAT to WASM
            compile_result = subprocess.run(
                ["wat2wasm", wat_path, "-o", wasm_path],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if compile_result.returncode != 0:
                return WasmResult(
                    exit_code=1,
                    stdout="",
                    stderr=f"WAT compilation failed: {compile_result.stderr}",
                    duration_ms=0,
                    memory_used_bytes=0,
                    runtime_version="unknown",
                    plugins_loaded=[],
                    trace_id=trace_id,
                )

            # Execute compiled WASM
            return self.execute(wasm_path, args=args, trace_id=trace_id)

        finally:
            # Cleanup temp files
            try:
                os.unlink(wat_path)
                if os.path.exists(wasm_path):
                    os.unlink(wasm_path)
            except Exception:
                pass

    def infer(
        self,
        model_path: str,
        prompt: str,
        backend: str = "ggml",
        max_tokens: int = 256,
        temperature: float = 0.7,
        trace_id: Optional[str] = None,
    ) -> InferenceResult:
        """
        Run neural inference via wasi_nn.

        Requires:
        - wasi_nn plugin
        - GGML or TensorFlow Lite backend
        - Pre-compiled inference WASM module

        Args:
            model_path: Path to model file (GGUF for GGML)
            prompt: Input prompt
            backend: Inference backend ("ggml" or "tensorflowlite")
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            trace_id: Trace ID for observability

        Returns:
            InferenceResult with generated output
        """
        start_time = time.monotonic()
        trace_id = trace_id or str(uuid.uuid4())

        # Validate wasi_nn availability
        if not WasmEdgeDetector.has_wasi_nn():
            return InferenceResult(
                output="",
                tokens_generated=0,
                duration_ms=0,
                tokens_per_second=0.0,
                model_name=Path(model_path).name,
                backend=backend,
                trace_id=trace_id,
            )

        # Check for inference WASM module
        # This would be a pre-compiled module that uses wasi_nn
        inference_wasm = os.environ.get(
            "WASMEDGE_INFERENCE_WASM",
            "/usr/local/share/wasmedge/llama-simple.wasm"
        )

        if not Path(inference_wasm).exists():
            # Fall back to dry-run
            duration_ms = int((time.monotonic() - start_time) * 1000)
            return InferenceResult(
                output=json.dumps({
                    "status": "dry_run",
                    "reason": f"Inference WASM not found: {inference_wasm}",
                    "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    "capabilities": WasmEdgeDetector.get_capabilities(),
                }),
                tokens_generated=0,
                duration_ms=duration_ms,
                tokens_per_second=0.0,
                model_name=Path(model_path).name,
                backend=backend,
                trace_id=trace_id,
            )

        # Build command with wasi_nn plugin
        cmd = [
            "wasmedge",
            "--dir", f".:{Path(model_path).parent}",
            "--nn-preload", f"default:{backend}:{model_path}",
        ]

        # Add inference WASM and arguments
        cmd.extend([
            inference_wasm,
            "--model-alias", "default",
            "--prompt", prompt,
            "--max-tokens", str(max_tokens),
            "--temp", str(temperature),
        ])

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # Inference can be slow
            )
            duration_ms = int((time.monotonic() - start_time) * 1000)

            # Parse output (format depends on inference WASM)
            output = proc.stdout.strip()

            # Estimate tokens (rough approximation)
            tokens_generated = len(output.split())
            tokens_per_second = (
                tokens_generated / (duration_ms / 1000)
                if duration_ms > 0 else 0.0
            )

            return InferenceResult(
                output=output,
                tokens_generated=tokens_generated,
                duration_ms=duration_ms,
                tokens_per_second=tokens_per_second,
                model_name=Path(model_path).name,
                backend=backend,
                trace_id=trace_id,
            )
        except subprocess.TimeoutExpired:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            return InferenceResult(
                output="",
                tokens_generated=0,
                duration_ms=duration_ms,
                tokens_per_second=0.0,
                model_name=Path(model_path).name,
                backend=backend,
                trace_id=trace_id,
            )
        except Exception as e:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            return InferenceResult(
                output=f"Error: {e}",
                tokens_generated=0,
                duration_ms=duration_ms,
                tokens_per_second=0.0,
                model_name=Path(model_path).name,
                backend=backend,
                trace_id=trace_id,
            )

    def compile_to_aot(
        self,
        wasm_path: str,
        output_path: Optional[str] = None,
    ) -> tuple[bool, str]:
        """
        Compile WASM to AOT (Ahead-of-Time) format for faster execution.

        Args:
            wasm_path: Path to .wasm file
            output_path: Output path (defaults to .so extension)

        Returns:
            Tuple of (success, output_path_or_error)
        """
        wasm_file = Path(wasm_path)
        if not wasm_file.exists():
            return (False, f"WASM file not found: {wasm_path}")

        out_path = output_path or str(wasm_file.with_suffix(".so"))

        cmd = ["wasmedge", "compile", str(wasm_file), out_path]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                return (True, out_path)
            else:
                return (False, result.stderr)
        except Exception as e:
            return (False, str(e))

    def get_capabilities(self) -> Dict[str, Any]:
        """Return runtime capabilities report."""
        caps = WasmEdgeDetector.get_capabilities()
        caps["dry_run"] = self.dry_run
        caps["config"] = {
            "max_memory_pages": self.config.max_memory_pages,
            "timeout_ms": self.config.timeout_ms,
            "simd_enabled": self.config.enable_simd,
            "threads_enabled": self.config.enable_threads,
        }
        return caps


class AgentSandbox:
    """
    High-level agent sandbox combining ContainerExecutor and WasmEdgeRuntime.

    Provides unified interface for isolated tool execution.
    """

    def __init__(self):
        self.wasm_runtime = WasmEdgeRuntime()
        # Import container executor
        try:
            from container_executor import ContainerExecutor, IsolationLevel
            self.container_executor = ContainerExecutor()
            self.has_containers = True
        except ImportError:
            self.has_containers = False
            self.container_executor = None

    def execute_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        execution_mode: str = "auto",
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a tool in the appropriate sandbox.

        Args:
            tool_name: Name of tool to execute
            tool_input: Tool input parameters
            execution_mode: "wasm", "container", or "auto"
            trace_id: Trace ID for observability

        Returns:
            Tool execution result
        """
        trace_id = trace_id or str(uuid.uuid4())

        # Auto-select based on tool type
        if execution_mode == "auto":
            # WASM for lightweight, deterministic tools
            # Containers for complex tools needing full OS
            if tool_name in ["hash", "parse_json", "format", "validate"]:
                execution_mode = "wasm"
            else:
                execution_mode = "container"

        if execution_mode == "wasm":
            return self._execute_wasm_tool(tool_name, tool_input, trace_id)
        else:
            return self._execute_container_tool(tool_name, tool_input, trace_id)

    def _execute_wasm_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        trace_id: str,
    ) -> Dict[str, Any]:
        """Execute tool via WASM runtime."""
        # Look for pre-compiled tool WASM
        tool_wasm = Path(f"/usr/local/share/pluribus/tools/{tool_name}.wasm")

        if not tool_wasm.exists():
            return {
                "success": False,
                "error": f"WASM tool not found: {tool_name}",
                "trace_id": trace_id,
            }

        result = self.wasm_runtime.execute(
            str(tool_wasm),
            args=[json.dumps(tool_input)],
            trace_id=trace_id,
        )

        try:
            output = json.loads(result.stdout) if result.stdout else {}
        except json.JSONDecodeError:
            output = {"raw": result.stdout}

        return {
            "success": result.exit_code == 0,
            "output": output,
            "stderr": result.stderr,
            "duration_ms": result.duration_ms,
            "trace_id": trace_id,
        }

    def _execute_container_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        trace_id: str,
    ) -> Dict[str, Any]:
        """Execute tool via container runtime."""
        if not self.has_containers or not self.container_executor:
            return {
                "success": False,
                "error": "Container executor not available",
                "trace_id": trace_id,
            }

        # Generic tool execution pattern
        result = self.container_executor.execute_code(
            f"import json; print(json.dumps({{'tool': '{tool_name}', 'input': {json.dumps(tool_input)}}}))",
            language="python",
            trace_id=trace_id,
        )

        try:
            output = json.loads(result.stdout) if result.stdout else {}
        except json.JSONDecodeError:
            output = {"raw": result.stdout}

        return {
            "success": result.exit_code == 0,
            "output": output,
            "stderr": result.stderr,
            "duration_ms": result.duration_ms,
            "trace_id": trace_id,
        }

    def get_capabilities(self) -> Dict[str, Any]:
        """Return combined sandbox capabilities."""
        caps = {
            "wasm": self.wasm_runtime.get_capabilities(),
            "containers": (
                self.container_executor.get_capabilities()
                if self.has_containers else {"available": False}
            ),
        }
        return caps


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="WasmEdge runtime for sandboxed WASM execution"
    )
    parser.add_argument("--wasm", help="WASM file to execute")
    parser.add_argument("--wat", help="WAT code to compile and execute")
    parser.add_argument("--args", nargs="*", help="Arguments for WASM program")
    parser.add_argument("--capabilities", action="store_true", help="Show capabilities")
    parser.add_argument("--infer", help="Model path for neural inference")
    parser.add_argument("--prompt", help="Prompt for inference")
    parser.add_argument("--compile-aot", help="Compile WASM to AOT")
    args = parser.parse_args()

    runtime = WasmEdgeRuntime()

    if args.capabilities:
        caps = runtime.get_capabilities()
        print(json.dumps(caps, indent=2))
        return

    if args.compile_aot:
        success, result = runtime.compile_to_aot(args.compile_aot)
        if success:
            print(f"AOT compiled: {result}")
        else:
            print(f"Compilation failed: {result}")
            sys.exit(1)
        return

    if args.infer and args.prompt:
        result = runtime.infer(
            model_path=args.infer,
            prompt=args.prompt,
            trace_id=str(uuid.uuid4()),
        )
        print(f"Output: {result.output}")
        print(f"Tokens: {result.tokens_generated}")
        print(f"Speed: {result.tokens_per_second:.2f} tok/s")
        return

    if args.wat:
        result = runtime.execute_wat(
            args.wat,
            args=args.args,
            trace_id=str(uuid.uuid4()),
        )
    elif args.wasm:
        result = runtime.execute(
            args.wasm,
            args=args.args,
            trace_id=str(uuid.uuid4()),
        )
    else:
        # Demo: capabilities only
        caps = runtime.get_capabilities()
        print(json.dumps(caps, indent=2))
        return

    print(f"Exit Code: {result.exit_code}")
    print(f"Duration: {result.duration_ms}ms")
    print(f"Runtime: {result.runtime_version}")
    print(f"Stdout: {result.stdout}")
    if result.stderr:
        print(f"Stderr: {result.stderr}")


if __name__ == "__main__":
    main()
