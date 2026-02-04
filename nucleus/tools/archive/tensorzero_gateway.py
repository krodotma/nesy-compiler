#!/usr/bin/env python3
"""
TensorZero Gateway Integration
==============================

Full router integration for TensorZero unified observability/routing gateway.
Provides:
- Inference routing through TensorZero
- Observability metrics emission to bus
- Experiment tracking via TensorZero API
- Feedback loop integration

TensorZero Reference: https://github.com/tensorzero/tensorzero

Usage:
    # Query inference
    python3 tensorzero_gateway.py infer --prompt "Summarize..."

    # Record feedback
    python3 tensorzero_gateway.py feedback --inference-id <id> --score 0.9

    # List experiments
    python3 tensorzero_gateway.py experiments

    # Emit metrics daemon mode
    python3 tensorzero_gateway.py daemon --emit-bus
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or getpass.getuser()


@dataclass
class TensorZeroConfig:
    """TensorZero gateway configuration."""
    gateway_url: str
    api_key: str | None = None
    default_function: str = "generate"
    timeout_s: float = 60.0
    retry_count: int = 2
    emit_bus: bool = True


@dataclass
class InferenceResult:
    """Result from TensorZero inference."""
    inference_id: str
    content: str
    model: str
    variant: str | None
    latency_ms: float
    input_tokens: int
    output_tokens: int
    raw_response: dict


@dataclass
class FeedbackResult:
    """Result from feedback submission."""
    feedback_id: str
    inference_id: str
    score: float
    metric_name: str
    accepted: bool


def emit_bus(bus_dir: str | None, *, topic: str, kind: str, level: str, actor: str, data: dict, trace_id: str | None = None) -> None:
    """Emit event to Pluribus bus."""
    if not bus_dir:
        return
    tool = Path(__file__).with_name("agent_bus.py")
    if not tool.exists():
        return
    cmd = [
        sys.executable,
        str(tool),
        "--bus-dir",
        bus_dir,
        "pub",
        "--topic",
        topic,
        "--kind",
        kind,
        "--level",
        level,
        "--actor",
        actor,
        "--data",
        json.dumps(data, ensure_ascii=False),
    ]
    if trace_id:
        cmd.extend(["--trace-id", trace_id])
    subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def load_config() -> TensorZeroConfig:
    """Load TensorZero configuration from environment."""
    gateway_url = os.environ.get("TENSORZERO_GATEWAY_URL", "").strip()
    if not gateway_url:
        gateway_url = os.environ.get("TENSORZERO_URL", "http://localhost:3000").strip()

    api_key = os.environ.get("TENSORZERO_API_KEY", "").strip() or None
    default_function = os.environ.get("TENSORZERO_DEFAULT_FUNCTION", "generate").strip()
    timeout_s = float(os.environ.get("TENSORZERO_TIMEOUT_S", "60"))
    retry_count = int(os.environ.get("TENSORZERO_RETRY_COUNT", "2"))
    emit_bus = os.environ.get("TENSORZERO_EMIT_BUS", "1").strip().lower() not in {"0", "false", "no"}

    return TensorZeroConfig(
        gateway_url=gateway_url,
        api_key=api_key,
        default_function=default_function,
        timeout_s=timeout_s,
        retry_count=retry_count,
        emit_bus=emit_bus,
    )


def _http_request(
    url: str,
    *,
    method: str = "POST",
    headers: dict[str, str] | None = None,
    data: dict | None = None,
    timeout: float = 60.0,
) -> tuple[int, dict | str]:
    """Make HTTP request using urllib (no external deps)."""
    import urllib.request
    import urllib.error

    headers = headers or {}
    headers.setdefault("Content-Type", "application/json")

    body = json.dumps(data).encode("utf-8") if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8")
            return e.code, {"error": body}
        except Exception:
            return e.code, {"error": str(e)}
    except Exception as e:
        return 0, {"error": str(e)}


class TensorZeroGateway:
    """TensorZero gateway client with full observability integration."""

    def __init__(self, config: TensorZeroConfig | None = None, bus_dir: str | None = None):
        self.config = config or load_config()
        self.bus_dir = bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
        self.actor = default_actor()
        self._metrics_buffer: list[dict] = []

    def _emit(self, topic: str, kind: str, level: str, data: dict, trace_id: str | None = None) -> None:
        """Emit event to bus if configured."""
        if self.config.emit_bus and self.bus_dir:
            emit_bus(self.bus_dir, topic=topic, kind=kind, level=level, actor=self.actor, data=data, trace_id=trace_id)

    def _build_headers(self) -> dict[str, str]:
        """Build request headers."""
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def health_check(self) -> bool:
        """Check if TensorZero gateway is healthy."""
        url = f"{self.config.gateway_url}/health"
        try:
            status, _ = _http_request(url, method="GET", timeout=5.0)
            return status == 200
        except Exception:
            return False

    def infer(
        self,
        prompt: str,
        *,
        function_name: str | None = None,
        model: str | None = None,
        variant: str | None = None,
        episode_id: str | None = None,
        trace_id: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> InferenceResult:
        """
        Run inference through TensorZero gateway.

        TensorZero routes the request based on:
        - function_name: Which function to call (maps to model variants)
        - variant: Specific variant override
        - Routing rules defined in gateway config
        """
        function_name = function_name or self.config.default_function
        url = f"{self.config.gateway_url}/inference"

        payload: dict[str, Any] = {
            "function_name": function_name,
            "input": {
                "messages": [{"role": "user", "content": prompt}]
            },
        }

        if variant:
            payload["variant_name"] = variant
        if episode_id:
            payload["episode_id"] = episode_id
        if tags:
            payload["tags"] = tags

        t0 = time.perf_counter()
        self._emit(
            "tensorzero.infer.start",
            kind="metric",
            level="info",
            data={
                "function": function_name,
                "variant": variant,
                "model": model,
                "prompt_len": len(prompt),
            },
            trace_id=trace_id,
        )

        status, resp = _http_request(
            url,
            method="POST",
            headers=self._build_headers(),
            data=payload,
            timeout=self.config.timeout_s,
        )

        latency_ms = (time.perf_counter() - t0) * 1000

        if status != 200 or not isinstance(resp, dict):
            error_msg = resp.get("error", str(resp)) if isinstance(resp, dict) else str(resp)
            self._emit(
                "tensorzero.infer.error",
                kind="metric",
                level="error",
                data={
                    "function": function_name,
                    "status": status,
                    "error": error_msg[:500],
                    "latency_ms": latency_ms,
                },
                trace_id=trace_id,
            )
            raise RuntimeError(f"TensorZero inference failed: {error_msg}")

        # Parse TensorZero response format
        # TensorZero returns: {inference_id, content, model_name, variant_name, usage}
        inference_id = resp.get("inference_id", str(uuid.uuid4()))

        # Content can be in different places based on TensorZero version
        content = ""
        if "content" in resp:
            content_raw = resp["content"]
            if isinstance(content_raw, list) and content_raw:
                # List of content blocks
                content = content_raw[0].get("text", "") if isinstance(content_raw[0], dict) else str(content_raw[0])
            elif isinstance(content_raw, str):
                content = content_raw
        elif "output" in resp:
            content = str(resp["output"])
        elif "choices" in resp:
            # OpenAI-compatible format
            choices = resp.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")

        model_name = resp.get("model_name", resp.get("model", model or "unknown"))
        variant_name = resp.get("variant_name", variant)

        usage = resp.get("usage", {})
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))

        result = InferenceResult(
            inference_id=inference_id,
            content=content,
            model=model_name,
            variant=variant_name,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw_response=resp,
        )

        self._emit(
            "tensorzero.infer.complete",
            kind="metric",
            level="info",
            data={
                "inference_id": inference_id,
                "function": function_name,
                "model": model_name,
                "variant": variant_name,
                "latency_ms": latency_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
            trace_id=trace_id,
        )

        # Buffer for aggregation
        self._metrics_buffer.append({
            "ts": time.time(),
            "inference_id": inference_id,
            "function": function_name,
            "model": model_name,
            "latency_ms": latency_ms,
            "tokens": input_tokens + output_tokens,
        })

        return result

    def feedback(
        self,
        inference_id: str,
        score: float,
        *,
        metric_name: str = "quality",
        comment: str | None = None,
        trace_id: str | None = None,
    ) -> FeedbackResult:
        """
        Submit feedback for an inference result.

        This enables the RL feedback loop in TensorZero for:
        - A/B testing variant selection
        - Model quality tracking
        - Automated variant promotion
        """
        url = f"{self.config.gateway_url}/feedback"

        payload: dict[str, Any] = {
            "inference_id": inference_id,
            "metric_name": metric_name,
            "value": score,
        }
        if comment:
            payload["comment"] = comment

        status, resp = _http_request(
            url,
            method="POST",
            headers=self._build_headers(),
            data=payload,
            timeout=10.0,
        )

        accepted = status == 200 or status == 201
        feedback_id = resp.get("feedback_id", str(uuid.uuid4())) if isinstance(resp, dict) else str(uuid.uuid4())

        result = FeedbackResult(
            feedback_id=feedback_id,
            inference_id=inference_id,
            score=score,
            metric_name=metric_name,
            accepted=accepted,
        )

        self._emit(
            "tensorzero.feedback",
            kind="metric",
            level="info" if accepted else "warn",
            data={
                "feedback_id": feedback_id,
                "inference_id": inference_id,
                "metric": metric_name,
                "score": score,
                "accepted": accepted,
            },
            trace_id=trace_id,
        )

        return result

    def list_experiments(self) -> list[dict]:
        """List active experiments from TensorZero."""
        url = f"{self.config.gateway_url}/experiments"
        status, resp = _http_request(url, method="GET", headers=self._build_headers(), timeout=10.0)

        if status != 200:
            return []

        experiments = resp.get("experiments", resp) if isinstance(resp, dict) else []
        if not isinstance(experiments, list):
            experiments = []

        self._emit(
            "tensorzero.experiments.list",
            kind="metric",
            level="info",
            data={"count": len(experiments)},
        )

        return experiments

    def get_metrics_summary(self) -> dict:
        """Get summary of buffered metrics."""
        if not self._metrics_buffer:
            return {"count": 0, "avg_latency_ms": 0, "total_tokens": 0}

        count = len(self._metrics_buffer)
        avg_latency = sum(m["latency_ms"] for m in self._metrics_buffer) / count
        total_tokens = sum(m.get("tokens", 0) for m in self._metrics_buffer)

        return {
            "count": count,
            "avg_latency_ms": round(avg_latency, 2),
            "total_tokens": total_tokens,
            "buffer_since": self._metrics_buffer[0]["ts"] if self._metrics_buffer else None,
        }

    def flush_metrics(self) -> None:
        """Flush buffered metrics to bus."""
        if not self._metrics_buffer:
            return

        summary = self.get_metrics_summary()
        self._emit(
            "tensorzero.metrics.flush",
            kind="metric",
            level="info",
            data=summary,
        )
        self._metrics_buffer.clear()


def cmd_infer(args: argparse.Namespace) -> int:
    """Run inference command."""
    gateway = TensorZeroGateway(bus_dir=args.bus_dir)

    if not gateway.health_check():
        sys.stderr.write(f"TensorZero gateway not reachable at {gateway.config.gateway_url}\n")
        return 1

    try:
        result = gateway.infer(
            args.prompt,
            function_name=args.function,
            model=args.model,
            variant=args.variant,
            trace_id=args.trace_id,
        )

        if args.json:
            sys.stdout.write(json.dumps(asdict(result), ensure_ascii=False, default=str) + "\n")
        else:
            sys.stdout.write(result.content + "\n")

        return 0
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1


def cmd_feedback(args: argparse.Namespace) -> int:
    """Submit feedback command."""
    gateway = TensorZeroGateway(bus_dir=args.bus_dir)

    try:
        result = gateway.feedback(
            args.inference_id,
            args.score,
            metric_name=args.metric,
            comment=args.comment,
            trace_id=args.trace_id,
        )

        if args.json:
            sys.stdout.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
        else:
            status = "accepted" if result.accepted else "rejected"
            sys.stdout.write(f"Feedback {result.feedback_id} {status}\n")

        return 0 if result.accepted else 1
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1


def cmd_experiments(args: argparse.Namespace) -> int:
    """List experiments command."""
    gateway = TensorZeroGateway(bus_dir=args.bus_dir)

    experiments = gateway.list_experiments()

    if args.json:
        sys.stdout.write(json.dumps(experiments, ensure_ascii=False, indent=2) + "\n")
    else:
        if not experiments:
            sys.stdout.write("No active experiments\n")
        else:
            for exp in experiments:
                name = exp.get("name", exp.get("id", "unknown"))
                status = exp.get("status", "unknown")
                sys.stdout.write(f"- {name}: {status}\n")

    return 0


def cmd_daemon(args: argparse.Namespace) -> int:
    """Run metrics daemon."""
    gateway = TensorZeroGateway(bus_dir=args.bus_dir)
    actor = default_actor()

    emit_bus(
        args.bus_dir,
        topic="tensorzero.daemon.start",
        kind="log",
        level="info",
        actor=actor,
        data={"gateway_url": gateway.config.gateway_url, "interval_s": args.interval},
    )

    sys.stdout.write(f"TensorZero metrics daemon started (interval={args.interval}s)\n")

    while True:
        try:
            # Check gateway health
            healthy = gateway.health_check()

            emit_bus(
                args.bus_dir,
                topic="tensorzero.health",
                kind="metric",
                level="info" if healthy else "warn",
                actor=actor,
                data={
                    "healthy": healthy,
                    "gateway_url": gateway.config.gateway_url,
                    "ts": now_iso_utc(),
                },
            )

            # Fetch and emit experiment status
            if healthy:
                experiments = gateway.list_experiments()
                emit_bus(
                    args.bus_dir,
                    topic="tensorzero.experiments.status",
                    kind="metric",
                    level="info",
                    actor=actor,
                    data={
                        "count": len(experiments),
                        "experiments": experiments[:10],  # Limit to avoid huge events
                    },
                )

            # Flush any buffered metrics
            gateway.flush_metrics()

        except Exception as e:
            emit_bus(
                args.bus_dir,
                topic="tensorzero.daemon.error",
                kind="log",
                level="error",
                actor=actor,
                data={"error": str(e)},
            )

        time.sleep(args.interval)


def cmd_health(args: argparse.Namespace) -> int:
    """Check gateway health."""
    gateway = TensorZeroGateway(bus_dir=args.bus_dir)
    healthy = gateway.health_check()

    if args.json:
        sys.stdout.write(json.dumps({"healthy": healthy, "gateway_url": gateway.config.gateway_url}) + "\n")
    else:
        status = "healthy" if healthy else "unhealthy"
        sys.stdout.write(f"TensorZero gateway: {status}\n")

    return 0 if healthy else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tensorzero_gateway.py",
        description="TensorZero gateway integration with full observability",
    )
    p.add_argument("--bus-dir", default=None, help="Bus directory (or PLURIBUS_BUS_DIR)")

    sub = p.add_subparsers(dest="cmd", required=True)

    # infer
    infer_p = sub.add_parser("infer", help="Run inference through TensorZero")
    infer_p.add_argument("--prompt", required=True, help="Prompt to send")
    infer_p.add_argument("--function", default=None, help="TensorZero function name")
    infer_p.add_argument("--model", default=None, help="Model hint")
    infer_p.add_argument("--variant", default=None, help="Specific variant to use")
    infer_p.add_argument("--trace-id", default=None, help="Trace ID for correlation")
    infer_p.add_argument("--json", action="store_true", help="Output as JSON")
    infer_p.set_defaults(func=cmd_infer)

    # feedback
    fb_p = sub.add_parser("feedback", help="Submit feedback for inference")
    fb_p.add_argument("--inference-id", required=True, help="Inference ID to rate")
    fb_p.add_argument("--score", type=float, required=True, help="Score (0.0-1.0)")
    fb_p.add_argument("--metric", default="quality", help="Metric name")
    fb_p.add_argument("--comment", default=None, help="Optional comment")
    fb_p.add_argument("--trace-id", default=None, help="Trace ID for correlation")
    fb_p.add_argument("--json", action="store_true", help="Output as JSON")
    fb_p.set_defaults(func=cmd_feedback)

    # experiments
    exp_p = sub.add_parser("experiments", help="List active experiments")
    exp_p.add_argument("--json", action="store_true", help="Output as JSON")
    exp_p.set_defaults(func=cmd_experiments)

    # daemon
    daemon_p = sub.add_parser("daemon", help="Run metrics daemon")
    daemon_p.add_argument("--interval", type=float, default=30.0, help="Poll interval seconds")
    daemon_p.set_defaults(func=cmd_daemon)

    # health
    health_p = sub.add_parser("health", help="Check gateway health")
    health_p.add_argument("--json", action="store_true", help="Output as JSON")
    health_p.set_defaults(func=cmd_health)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
