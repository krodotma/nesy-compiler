#!/usr/bin/env python3
"""
manim_renderer.py - ManimCE Animation Rendering Backend

Provides server-side rendering of Manim (Community Edition) scenes for
the Pluribus dashboard. Supports:
- Scene code execution with sandboxing
- Video/GIF output generation
- Preview frame extraction
- Bus event integration for render progress

Usage:
    python3 manim_renderer.py render --code "class MyScene(Scene): ..." --output /tmp/out.mp4
    python3 manim_renderer.py server --port 9210

Reference: https://docs.manim.community/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

# Try importing agent_bus for event emission
try:
    from agent_bus import resolve_bus_paths, emit_event
except ImportError:
    resolve_bus_paths = None  # type: ignore
    emit_event = None  # type: ignore


# ============================================================================
# Configuration
# ============================================================================

MANIM_QUALITY_PRESETS = {
    "preview": {"quality": "l", "fps": 15, "pixel_height": 480},
    "low": {"quality": "l", "fps": 30, "pixel_height": 480},
    "medium": {"quality": "m", "fps": 30, "pixel_height": 720},
    "high": {"quality": "h", "fps": 60, "pixel_height": 1080},
    "4k": {"quality": "p", "fps": 60, "pixel_height": 2160},
}

DEFAULT_QUALITY = "preview"
RENDER_TIMEOUT_SECONDS = 300  # 5 minutes max per render
MANIM_MEDIA_DIR = Path(os.environ.get("MANIM_MEDIA_DIR", "/tmp/manim_media"))


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class RenderRequest:
    """Request to render a Manim scene."""
    code: str
    scene_name: str | None = None  # Auto-detect if not provided
    quality: str = DEFAULT_QUALITY
    output_format: str = "mp4"  # mp4, gif, png (last frame), webm
    output_path: str | None = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    extra_args: list[str] = field(default_factory=list)


@dataclass
class RenderResult:
    """Result of a Manim render operation."""
    success: bool
    request_id: str
    output_path: str | None = None
    preview_frame: str | None = None  # Base64 PNG of last frame
    duration_seconds: float = 0.0
    frame_count: int = 0
    error: str | None = None
    stderr: str | None = None


# ============================================================================
# Scene Detection
# ============================================================================

def detect_scene_names(code: str) -> list[str]:
    """
    Extract Scene class names from Manim code.
    Returns list of class names that inherit from Scene.
    """
    import re

    # Match class definitions that inherit from Scene or ThreeDScene
    pattern = r"class\s+(\w+)\s*\(\s*(?:\w+\.)?(?:Scene|ThreeDScene|MovingCameraScene|ZoomedScene)\s*\)"
    matches = re.findall(pattern, code)
    return matches


# ============================================================================
# Rendering Engine
# ============================================================================

def render_scene(request: RenderRequest) -> RenderResult:
    """
    Render a Manim scene from code.

    Creates a temporary Python file, invokes manim CLI, and returns the result.
    """
    start_time = time.time()
    request_id = request.request_id

    # Emit start event
    if emit_event and resolve_bus_paths:
        try:
            paths = resolve_bus_paths(None)
            emit_event(
                paths,
                topic="manim.render.started",
                kind="log",
                level="info",
                actor="manim-renderer",
                data={
                    "request_id": request_id,
                    "quality": request.quality,
                    "format": request.output_format,
                },
                trace_id=None,
                run_id=None,
                durable=False,
            )
        except Exception:
            pass  # Bus errors are non-fatal

    # Detect scene name if not provided
    scene_name = request.scene_name
    if not scene_name:
        detected = detect_scene_names(request.code)
        if not detected:
            return RenderResult(
                success=False,
                request_id=request_id,
                error="No Scene class found in code. Define a class inheriting from Scene.",
            )
        scene_name = detected[0]  # Use first scene

    # Create temporary directory for this render
    work_dir = Path(tempfile.mkdtemp(prefix=f"manim_{request_id[:8]}_"))
    script_path = work_dir / "scene.py"

    try:
        # Write scene code to file
        # Prepend imports if not present
        code = request.code
        if "from manim import" not in code and "import manim" not in code:
            code = "from manim import *\n\n" + code

        script_path.write_text(code, encoding="utf-8")

        # Build manim command
        quality_config = MANIM_QUALITY_PRESETS.get(request.quality, MANIM_QUALITY_PRESETS[DEFAULT_QUALITY])

        cmd = [
            sys.executable, "-m", "manim",
            str(script_path),
            scene_name,
            f"-q{quality_config['quality']}",
            "--media_dir", str(MANIM_MEDIA_DIR),
            "--disable_caching",
        ]

        # Add format-specific flags
        if request.output_format == "gif":
            cmd.append("--format=gif")
        elif request.output_format == "webm":
            cmd.append("--format=webm")
        elif request.output_format == "png":
            cmd.extend(["-s", "--format=png"])  # Save last frame

        # Add any extra arguments
        cmd.extend(request.extra_args)

        # Execute manim
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=RENDER_TIMEOUT_SECONDS,
            cwd=work_dir,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )

        duration = time.time() - start_time

        if proc.returncode != 0:
            return RenderResult(
                success=False,
                request_id=request_id,
                error=f"Manim render failed (exit {proc.returncode})",
                stderr=proc.stderr[-2000:] if proc.stderr else None,
                duration_seconds=duration,
            )

        # Find output file
        media_dir = MANIM_MEDIA_DIR / "videos" / "scene" / f"{quality_config['pixel_height']}p{quality_config['fps']}"

        # Search for the output file
        output_file = None
        extensions = {
            "mp4": [".mp4"],
            "gif": [".gif"],
            "webm": [".webm"],
            "png": [".png"],
        }

        for ext in extensions.get(request.output_format, [".mp4"]):
            pattern = f"{scene_name}{ext}"
            matches = list(media_dir.glob(pattern)) if media_dir.exists() else []
            if matches:
                output_file = matches[0]
                break

        # Also check images directory for PNG
        if not output_file and request.output_format == "png":
            images_dir = MANIM_MEDIA_DIR / "images" / "scene"
            if images_dir.exists():
                matches = list(images_dir.glob(f"{scene_name}*.png"))
                if matches:
                    output_file = sorted(matches)[-1]  # Latest

        if not output_file or not output_file.exists():
            return RenderResult(
                success=False,
                request_id=request_id,
                error=f"Output file not found. Expected in {media_dir}",
                stderr=proc.stderr[-1000:] if proc.stderr else None,
                duration_seconds=duration,
            )

        # Copy to requested output path if specified
        final_path = str(output_file)
        if request.output_path:
            final_dir = Path(request.output_path).parent
            final_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_file, request.output_path)
            final_path = request.output_path

        # Generate preview frame (base64 PNG)
        preview_frame = None
        if request.output_format == "mp4" and output_file.exists():
            try:
                preview_path = work_dir / "preview.png"
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-i", str(output_file),
                    "-vf", "select=eq(n\\,0)",
                    "-frames:v", "1",
                    str(preview_path),
                ]
                subprocess.run(ffmpeg_cmd, capture_output=True, timeout=30)
                if preview_path.exists():
                    import base64
                    preview_frame = base64.b64encode(preview_path.read_bytes()).decode("ascii")
            except Exception:
                pass  # Preview is optional

        # Emit completion event
        if emit_event and resolve_bus_paths:
            try:
                paths = resolve_bus_paths(None)
                emit_event(
                    paths,
                    topic="manim.render.complete",
                    kind="response",
                    level="info",
                    actor="manim-renderer",
                    data={
                        "request_id": request_id,
                        "output_path": final_path,
                        "duration_seconds": duration,
                        "quality": request.quality,
                    },
                    trace_id=None,
                    run_id=None,
                    durable=False,
                )
            except Exception:
                pass

        return RenderResult(
            success=True,
            request_id=request_id,
            output_path=final_path,
            preview_frame=preview_frame,
            duration_seconds=duration,
        )

    except subprocess.TimeoutExpired:
        return RenderResult(
            success=False,
            request_id=request_id,
            error=f"Render timeout ({RENDER_TIMEOUT_SECONDS}s exceeded)",
            duration_seconds=time.time() - start_time,
        )
    except Exception as e:
        return RenderResult(
            success=False,
            request_id=request_id,
            error=str(e),
            duration_seconds=time.time() - start_time,
        )
    finally:
        # Cleanup temp directory
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass


# ============================================================================
# HTTP Server (for dashboard integration)
# ============================================================================

def run_server(port: int = 9210) -> None:
    """
    Run HTTP server for Manim rendering requests.

    Endpoints:
        POST /render - Submit render request
        GET /status/<request_id> - Check render status
        GET /health - Health check
    """
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import threading

    # In-memory job tracking
    jobs: dict[str, dict[str, Any]] = {}
    jobs_lock = threading.Lock()

    class ManimHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            # Suppress default logging
            pass

        def send_json(self, status: int, data: dict[str, Any]) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self) -> None:
            if self.path == "/health":
                # Check if manim is available
                try:
                    result = subprocess.run(
                        [sys.executable, "-c", "import manim; print(manim.__version__)"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    version = result.stdout.strip() if result.returncode == 0 else None
                    self.send_json(200, {
                        "status": "healthy" if version else "degraded",
                        "manim_version": version,
                        "jobs_pending": sum(1 for j in jobs.values() if j.get("status") == "pending"),
                    })
                except Exception as e:
                    self.send_json(503, {"status": "unhealthy", "error": str(e)})
                return

            if self.path.startswith("/status/"):
                request_id = self.path.split("/")[-1]
                with jobs_lock:
                    job = jobs.get(request_id)
                if not job:
                    self.send_json(404, {"error": "Job not found"})
                else:
                    self.send_json(200, job)
                return

            self.send_json(404, {"error": "Not found"})

        def do_POST(self) -> None:
            if self.path != "/render":
                self.send_json(404, {"error": "Not found"})
                return

            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)

                if "code" not in data:
                    self.send_json(400, {"error": "Missing 'code' field"})
                    return

                request = RenderRequest(
                    code=data["code"],
                    scene_name=data.get("scene_name"),
                    quality=data.get("quality", DEFAULT_QUALITY),
                    output_format=data.get("output_format", "mp4"),
                    extra_args=data.get("extra_args", []),
                )

                # Register job
                with jobs_lock:
                    jobs[request.request_id] = {
                        "request_id": request.request_id,
                        "status": "pending",
                        "created_at": time.time(),
                    }

                # Start render in background
                def do_render() -> None:
                    result = render_scene(request)
                    with jobs_lock:
                        jobs[request.request_id] = {
                            "request_id": request.request_id,
                            "status": "complete" if result.success else "failed",
                            "success": result.success,
                            "output_path": result.output_path,
                            "preview_frame": result.preview_frame,
                            "duration_seconds": result.duration_seconds,
                            "error": result.error,
                            "completed_at": time.time(),
                        }

                thread = threading.Thread(target=do_render, daemon=True)
                thread.start()

                self.send_json(202, {
                    "request_id": request.request_id,
                    "status": "pending",
                    "message": "Render started",
                })

            except json.JSONDecodeError as e:
                self.send_json(400, {"error": f"Invalid JSON: {e}"})
            except Exception as e:
                self.send_json(500, {"error": str(e)})

    server = HTTPServer(("0.0.0.0", port), ManimHandler)
    print(f"[manim-renderer] Server listening on http://0.0.0.0:{port}")
    print(f"[manim-renderer] Media directory: {MANIM_MEDIA_DIR}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[manim-renderer] Shutting down...")


# ============================================================================
# CLI
# ============================================================================

def cmd_render(args: argparse.Namespace) -> int:
    """Render a scene from code."""
    code = args.code

    # Read from file if --file is provided
    if args.file:
        code = Path(args.file).read_text(encoding="utf-8")

    if not code:
        print("Error: No code provided. Use --code or --file.", file=sys.stderr)
        return 1

    request = RenderRequest(
        code=code,
        scene_name=args.scene,
        quality=args.quality,
        output_format=args.format,
        output_path=args.output,
    )

    result = render_scene(request)

    if result.success:
        print(f"Render complete: {result.output_path}")
        print(f"Duration: {result.duration_seconds:.2f}s")
        return 0
    else:
        print(f"Render failed: {result.error}", file=sys.stderr)
        if result.stderr:
            print(f"stderr:\n{result.stderr}", file=sys.stderr)
        return 1


def cmd_server(args: argparse.Namespace) -> int:
    """Run HTTP render server."""
    run_server(port=args.port)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Check if ManimCE is installed and working."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import manim; print(f'ManimCE {manim.__version__}')"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            print(result.stdout.strip())
            # Test render
            test_code = """
class TestScene(Scene):
    def construct(self):
        circle = Circle()
        self.add(circle)
"""
            request = RenderRequest(code=test_code, quality="preview", output_format="png")
            test_result = render_scene(request)
            if test_result.success:
                print(f"Test render OK: {test_result.output_path}")
                return 0
            else:
                print(f"Test render failed: {test_result.error}", file=sys.stderr)
                return 1
        else:
            print(f"ManimCE not working: {result.stderr}", file=sys.stderr)
            return 1
    except Exception as e:
        print(f"ManimCE check failed: {e}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="manim_renderer.py",
        description="ManimCE rendering backend for Pluribus dashboard",
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # Render command
    render_p = subparsers.add_parser("render", help="Render a Manim scene")
    render_p.add_argument("--code", help="Scene code as string")
    render_p.add_argument("--file", "-f", help="Read code from file")
    render_p.add_argument("--scene", "-s", help="Scene class name (auto-detect if omitted)")
    render_p.add_argument("--quality", "-q", default=DEFAULT_QUALITY,
                          choices=list(MANIM_QUALITY_PRESETS.keys()),
                          help="Quality preset")
    render_p.add_argument("--format", default="mp4",
                          choices=["mp4", "gif", "webm", "png"],
                          help="Output format")
    render_p.add_argument("--output", "-o", help="Output file path")
    render_p.set_defaults(func=cmd_render)

    # Server command
    server_p = subparsers.add_parser("server", help="Run HTTP render server")
    server_p.add_argument("--port", "-p", type=int, default=9210,
                          help="Server port (default: 9210)")
    server_p.set_defaults(func=cmd_server)

    # Check command
    check_p = subparsers.add_parser("check", help="Check ManimCE installation")
    check_p.set_defaults(func=cmd_check)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
