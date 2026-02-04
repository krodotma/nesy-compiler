#!/usr/bin/env python3
"""GEN3C Video: 3D-Consistent Video Generation.

Wrapper for GEN3C model that generates 3D-consistent videos from sparse views.
Maintains geometric consistency across frames, suitable for camera fly-throughs
and scene exploration.

Hardware Requirements:
- GPU Memory: 12GB+ (CUDA)
- Generation Time: ~30s per 4-second clip

Pluribus Integration:
- Dashboard: Video preview widgets
- Creative Tools: Scene exploration assets
- Bus Events: sota.gen3c.request -> sota.gen3c.response

Fallback Mode:
- When GPU unavailable, generates placeholder video frames with motion blur effect.

Usage:
    # Generate video from views
    python3 gen3c_video.py generate --views /path/to/views/*.png --duration 4

    # Serve as HTTP endpoint
    python3 gen3c_video.py serve --port 9302

    # Daemon mode (bus-driven)
    python3 gen3c_video.py daemon
"""
from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

# ============================================================================
# Bus Integration
# ============================================================================

def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def resolve_bus_path() -> Path:
    """Resolve the pluribus bus events path."""
    default_bus = Path("/pluribus/.pluribus/bus")
    if not default_bus.exists():
        state_home = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
        default_bus = state_home / "nucleus" / "bus"
    bus_dir = Path(os.environ.get("PLURIBUS_BUS_DIR", str(default_bus)))
    bus_dir.mkdir(parents=True, exist_ok=True)
    return bus_dir / "events.ndjson"


def emit_bus_event(topic: str, kind: str, data: dict, level: str = "info") -> str:
    """Emit an event to the pluribus bus."""
    event_id = str(uuid.uuid4())
    event = {
        "id": event_id,
        "ts": time.time(),
        "iso": now_iso_utc(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": os.environ.get("PLURIBUS_ACTOR", "gen3c-video"),
        "data": data,
    }
    bus_path = resolve_bus_path()
    with bus_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
    return event_id


def tail_bus_events(topic_prefix: str, since_ts: float = 0) -> list[dict]:
    """Read recent bus events matching topic prefix."""
    events = []
    bus_path = resolve_bus_path()
    if not bus_path.exists():
        return events
    with bus_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("ts", 0) > since_ts and obj.get("topic", "").startswith(topic_prefix):
                    events.append(obj)
            except Exception:
                continue
    return events


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Gen3CRequest:
    """Request for GEN3C video generation."""
    view_paths: list[str]
    duration_seconds: float = 4.0
    fps: int = 24
    resolution: tuple[int, int] = (512, 512)
    camera_path: str = "orbit"  # orbit | linear | spline
    seed: int = -1

    def to_dict(self) -> dict:
        return {
            "view_paths": self.view_paths,
            "duration_seconds": self.duration_seconds,
            "fps": self.fps,
            "resolution": list(self.resolution),
            "camera_path": self.camera_path,
            "seed": self.seed,
        }


@dataclass
class Gen3CResult:
    """Result of GEN3C video generation."""
    request: Gen3CRequest
    video_path: str | None = None
    video_bytes: bytes | None = None
    frame_count: int = 0
    metadata: dict = field(default_factory=dict)
    generation_time_ms: float = 0.0
    status: str = "success"
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "request": self.request.to_dict(),
            "video_path": self.video_path,
            "video_bytes_b64": base64.b64encode(self.video_bytes).decode() if self.video_bytes else None,
            "frame_count": self.frame_count,
            "metadata": self.metadata,
            "generation_time_ms": self.generation_time_ms,
            "status": self.status,
            "error": self.error,
        }


# ============================================================================
# GPU Detection & Model Loading
# ============================================================================

_GPU_AVAILABLE: bool | None = None
_MODEL_LOADED: Any = None


def check_gpu_available() -> bool:
    """Check if CUDA GPU is available."""
    global _GPU_AVAILABLE
    if _GPU_AVAILABLE is not None:
        return _GPU_AVAILABLE
    try:
        import torch
        _GPU_AVAILABLE = torch.cuda.is_available()
    except ImportError:
        _GPU_AVAILABLE = False
    return _GPU_AVAILABLE


def load_gen3c_model():
    """Load GEN3C model (or return mock if unavailable)."""
    global _MODEL_LOADED
    if _MODEL_LOADED is not None:
        return _MODEL_LOADED

    if not check_gpu_available():
        print("[GEN3C] GPU not available. Using fallback mock mode.", file=sys.stderr)
        _MODEL_LOADED = "mock"
        return _MODEL_LOADED

    try:
        import torch

        # Attempt to load actual GEN3C model
        try:
            from gen3c import Gen3CModel  # type: ignore
            device = "cuda"
            model = Gen3CModel.from_pretrained("nvidia/gen3c-base")
            model.to(device)
            model.eval()
            _MODEL_LOADED = model
            print(f"[GEN3C] Model loaded on {device}.")
        except ImportError:
            print("[GEN3C] GEN3C package not found. Using fallback mock mode.", file=sys.stderr)
            _MODEL_LOADED = "mock"

    except Exception as e:
        print(f"[GEN3C] Model loading failed: {e}. Using fallback mock mode.", file=sys.stderr)
        _MODEL_LOADED = "mock"

    return _MODEL_LOADED


# ============================================================================
# Frame Generation (Real & Mock)
# ============================================================================

def generate_mock_frame(
    base_images: list,
    frame_idx: int,
    total_frames: int,
    width: int,
    height: int,
) -> bytes:
    """Generate a mock video frame with interpolation and motion blur."""
    try:
        from PIL import Image, ImageFilter, ImageDraw
        import numpy as np

        if not base_images:
            # Create a solid color frame with frame number
            arr = np.zeros((height, width, 3), dtype=np.uint8)
            # Gradient based on frame index
            progress = frame_idx / max(1, total_frames - 1)
            arr[:, :, 0] = int(50 + 150 * progress)  # R
            arr[:, :, 1] = int(50 + 100 * (1 - progress))  # G
            arr[:, :, 2] = 100
            img = Image.fromarray(arr)
        else:
            # Blend between input views based on frame position
            progress = frame_idx / max(1, total_frames - 1)
            view_idx = int(progress * (len(base_images) - 1))
            next_idx = min(view_idx + 1, len(base_images) - 1)
            blend_factor = (progress * (len(base_images) - 1)) % 1.0

            img1 = base_images[view_idx].resize((width, height))
            img2 = base_images[next_idx].resize((width, height))

            # Blend images
            img = Image.blend(img1, img2, blend_factor)

            # Add subtle motion blur for realism
            if frame_idx > 0 and frame_idx < total_frames - 1:
                img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

        # Add frame indicator
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"Frame {frame_idx + 1}/{total_frames}", fill=(255, 255, 255))
        draw.text((10, height - 25), "[GEN3C Mock Mode]", fill=(200, 200, 200))

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    except ImportError:
        # Return a tiny valid PNG
        return b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'


def frames_to_video(frame_paths: list[Path], output_path: Path, fps: int) -> bool:
    """Convert frames to video using ffmpeg."""
    if not frame_paths:
        return False

    # Create a file list for ffmpeg
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for fp in frame_paths:
            f.write(f"file '{fp}'\n")
            f.write(f"duration {1/fps}\n")
        list_file = f.name

    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_file,
            "-vf", f"fps={fps}",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        # ffmpeg not available, try alternative
        try:
            from PIL import Image
            import struct
            import zlib

            # Create a simple GIF as fallback
            images = [Image.open(fp) for fp in frame_paths[:50]]  # Limit for GIF
            if images:
                gif_path = output_path.with_suffix(".gif")
                images[0].save(
                    gif_path,
                    save_all=True,
                    append_images=images[1:],
                    duration=int(1000 / fps),
                    loop=0,
                )
                # Copy to expected output
                import shutil
                shutil.copy(gif_path, output_path)
                return True
        except Exception:
            pass
        return False
    finally:
        try:
            os.unlink(list_file)
        except Exception:
            pass


# ============================================================================
# Video Generation
# ============================================================================

def generate_video(request: Gen3CRequest) -> Gen3CResult:
    """Generate 3D-consistent video from sparse views.

    Args:
        request: Gen3CRequest with view paths and parameters

    Returns:
        Gen3CResult with video data
    """
    start_time = time.time()
    model = load_gen3c_model()

    # Load input views
    view_images = []
    try:
        from PIL import Image
        for vp in request.view_paths:
            if Path(vp).exists():
                view_images.append(Image.open(vp).convert("RGB"))
    except ImportError:
        pass

    if not view_images and request.view_paths:
        return Gen3CResult(
            request=request,
            status="error",
            error="Could not load input views (PIL not available or files not found)",
            generation_time_ms=(time.time() - start_time) * 1000,
        )

    # Calculate frame count
    total_frames = int(request.duration_seconds * request.fps)
    width, height = request.resolution

    # Create temporary directory for frames
    work_dir = Path(tempfile.mkdtemp(prefix="gen3c_"))

    if model == "mock":
        # Generate mock frames
        frame_paths = []
        for i in range(total_frames):
            frame_data = generate_mock_frame(view_images, i, total_frames, width, height)
            frame_path = work_dir / f"frame_{i:05d}.png"
            frame_path.write_bytes(frame_data)
            frame_paths.append(frame_path)

        # Combine frames to video
        video_path = work_dir / "output.mp4"
        success = frames_to_video(frame_paths, video_path, request.fps)

        if not success:
            # Return as frame sequence
            return Gen3CResult(
                request=request,
                video_path=str(work_dir),
                frame_count=total_frames,
                metadata={
                    "mode": "mock",
                    "format": "frame_sequence",
                    "frame_pattern": "frame_*.png",
                },
                generation_time_ms=(time.time() - start_time) * 1000,
                status="success",
            )

        video_bytes = video_path.read_bytes()

        return Gen3CResult(
            request=request,
            video_path=str(video_path),
            video_bytes=video_bytes,
            frame_count=total_frames,
            metadata={
                "mode": "mock",
                "format": "mp4",
                "width": width,
                "height": height,
                "fps": request.fps,
                "model": "gen3c-fallback",
            },
            generation_time_ms=(time.time() - start_time) * 1000,
            status="success",
        )

    # Real model inference
    try:
        import torch
        import numpy as np
        from PIL import Image

        # Prepare input tensors
        view_tensors = []
        for img in view_images:
            img_resized = img.resize(request.resolution)
            arr = np.array(img_resized) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
            view_tensors.append(tensor)

        views_batch = torch.cat(view_tensors, dim=0)
        device = next(model.parameters()).device
        views_batch = views_batch.to(device)

        # Generate camera path
        camera_poses = generate_camera_path(
            request.camera_path,
            total_frames,
            len(view_images),
        )

        # Generate frames
        frame_paths = []
        with torch.no_grad():
            for i, pose in enumerate(camera_poses):
                pose_tensor = torch.tensor(pose).unsqueeze(0).to(device)
                frame = model.render(views_batch, pose_tensor)

                # Convert to image
                frame_np = (frame[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                frame_img = Image.fromarray(frame_np)

                frame_path = work_dir / f"frame_{i:05d}.png"
                frame_img.save(frame_path)
                frame_paths.append(frame_path)

        # Combine frames to video
        video_path = work_dir / "output.mp4"
        success = frames_to_video(frame_paths, video_path, request.fps)

        video_bytes = video_path.read_bytes() if video_path.exists() else None

        return Gen3CResult(
            request=request,
            video_path=str(video_path),
            video_bytes=video_bytes,
            frame_count=total_frames,
            metadata={
                "mode": "gpu",
                "format": "mp4" if success else "frame_sequence",
                "width": width,
                "height": height,
                "fps": request.fps,
                "model": "gen3c-base",
            },
            generation_time_ms=(time.time() - start_time) * 1000,
            status="success",
        )

    except Exception as e:
        return Gen3CResult(
            request=request,
            status="error",
            error=str(e),
            generation_time_ms=(time.time() - start_time) * 1000,
        )


def generate_camera_path(path_type: str, num_frames: int, num_views: int) -> list:
    """Generate camera poses for video generation."""
    import math

    poses = []

    if path_type == "orbit":
        for i in range(num_frames):
            angle = 2 * math.pi * i / num_frames
            pose = [
                math.cos(angle), 0, math.sin(angle), 0,  # Camera position
                0, 1, 0, 0,
                -math.sin(angle), 0, math.cos(angle), 0,
                0, 0, 0, 1,
            ]
            poses.append(pose)

    elif path_type == "linear":
        for i in range(num_frames):
            t = i / max(1, num_frames - 1)
            pose = [
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, -2 + 4 * t,  # Move along Z
                0, 0, 0, 1,
            ]
            poses.append(pose)

    else:  # spline or default
        # Simple interpolation between views
        for i in range(num_frames):
            t = i / max(1, num_frames - 1)
            view_idx = t * (num_views - 1)
            pose = [
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, view_idx,
                0, 0, 0, 1,
            ]
            poses.append(pose)

    return poses


# ============================================================================
# CLI Commands
# ============================================================================

def cmd_generate(args: argparse.Namespace) -> int:
    """Generate video from views."""
    # Expand view paths (support glob patterns)
    view_paths = []
    for pattern in args.views:
        expanded = glob.glob(pattern)
        if expanded:
            view_paths.extend(sorted(expanded))
        elif Path(pattern).exists():
            view_paths.append(pattern)

    if not view_paths:
        print("[GEN3C] Error: No valid view images found.", file=sys.stderr)
        return 1

    print(f"[GEN3C] Input views: {len(view_paths)}")

    request = Gen3CRequest(
        view_paths=view_paths,
        duration_seconds=args.duration,
        fps=args.fps,
        resolution=(args.width, args.height),
        camera_path=args.camera,
        seed=args.seed,
    )

    result = generate_video(request)

    if args.emit_bus:
        # Don't include video_bytes in bus event (too large)
        event_data = result.to_dict()
        event_data.pop("video_bytes_b64", None)
        emit_bus_event(
            topic="sota.gen3c.response",
            kind="generation",
            data=event_data,
            level="info" if result.status == "success" else "error",
        )

    if result.status != "success":
        print(f"[GEN3C] Error: {result.error}", file=sys.stderr)
        return 1

    # Copy output if specified
    if args.output and result.video_path:
        output_path = Path(args.output)
        if result.video_bytes:
            output_path.write_bytes(result.video_bytes)
            print(f"[GEN3C] Video saved to: {output_path}")
        else:
            print(f"[GEN3C] Frames saved to: {result.video_path}")

    print(f"[GEN3C] Generation complete in {result.generation_time_ms:.1f}ms")
    print(f"[GEN3C] Frames: {result.frame_count}, Mode: {result.metadata.get('mode', 'unknown')}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Serve GEN3C generation as HTTP endpoint."""
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class Gen3CHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok", "gpu": check_gpu_available()}).encode())
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path.startswith("/generate"):
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len)

                try:
                    data = json.loads(body)

                    request = Gen3CRequest(
                        view_paths=data.get("view_paths", []),
                        duration_seconds=data.get("duration_seconds", 4.0),
                        fps=data.get("fps", 24),
                        resolution=tuple(data.get("resolution", [512, 512])),
                        camera_path=data.get("camera_path", "orbit"),
                        seed=data.get("seed", -1),
                    )

                    result = generate_video(request)

                    if args.emit_bus:
                        event_data = result.to_dict()
                        event_data.pop("video_bytes_b64", None)
                        emit_bus_event(
                            topic="sota.gen3c.response",
                            kind="generation",
                            data=event_data,
                        )

                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(result.to_dict()).encode())

                except Exception as e:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode())
            else:
                self.send_error(404)

        def log_message(self, format, *args):
            print(f"[GEN3C] {args[0]}")

    server = HTTPServer(("0.0.0.0", args.port), Gen3CHandler)
    print(f"[GEN3C] Serving on http://0.0.0.0:{args.port}")
    print(f"[GEN3C] GPU available: {check_gpu_available()}")

    emit_bus_event(
        topic="sota.gen3c.started",
        kind="service",
        data={"port": args.port, "gpu": check_gpu_available()},
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[GEN3C] Shutting down...")
    return 0


def cmd_daemon(args: argparse.Namespace) -> int:
    """Run as bus-driven daemon."""
    print("[GEN3C] Starting daemon mode...")
    print(f"[GEN3C] GPU available: {check_gpu_available()}")
    print("[GEN3C] Watching topic: sota.gen3c.request")

    emit_bus_event(
        topic="sota.gen3c.daemon.started",
        kind="service",
        data={"gpu": check_gpu_available()},
    )

    last_ts = time.time()

    while True:
        events = tail_bus_events("sota.gen3c.request", since_ts=last_ts)

        for event in events:
            last_ts = max(last_ts, event.get("ts", 0))
            data = event.get("data", {})
            req_id = data.get("req_id", event.get("id", str(uuid.uuid4())))

            view_paths = data.get("view_paths", [])
            if not view_paths:
                continue

            print(f"[GEN3C] Processing request {req_id}: {len(view_paths)} views")

            request = Gen3CRequest(
                view_paths=view_paths,
                duration_seconds=data.get("duration_seconds", 4.0),
                fps=data.get("fps", 24),
                resolution=tuple(data.get("resolution", [512, 512])),
                camera_path=data.get("camera_path", "orbit"),
                seed=data.get("seed", -1),
            )

            result = generate_video(request)

            event_data = result.to_dict()
            event_data.pop("video_bytes_b64", None)
            event_data["req_id"] = req_id

            emit_bus_event(
                topic="sota.gen3c.response",
                kind="generation",
                data=event_data,
                level="info" if result.status == "success" else "error",
            )

        time.sleep(1.0)


def cmd_test(args: argparse.Namespace) -> int:
    """Test video generation."""
    print("[GEN3C] Running test generation...")
    print(f"[GEN3C] GPU available: {check_gpu_available()}")

    # Create test views if none provided
    view_paths = []
    if args.views:
        view_paths = args.views
    else:
        try:
            from PIL import Image
            import numpy as np

            test_dir = Path(tempfile.mkdtemp(prefix="gen3c_test_"))

            # Create 3 simple test views with different colors
            for i, color in enumerate([(255, 100, 100), (100, 255, 100), (100, 100, 255)]):
                arr = np.full((256, 256, 3), color, dtype=np.uint8)
                img = Image.fromarray(arr)
                path = test_dir / f"view_{i}.png"
                img.save(path)
                view_paths.append(str(path))
                print(f"[GEN3C] Created test view: {path}")

        except ImportError:
            print("[GEN3C] PIL not available. Please provide --views.", file=sys.stderr)
            return 1

    request = Gen3CRequest(
        view_paths=view_paths,
        duration_seconds=2.0,
        fps=12,
        resolution=(256, 256),
        camera_path="orbit",
    )

    result = generate_video(request)

    print(f"[GEN3C] Status: {result.status}")
    print(f"[GEN3C] Generation time: {result.generation_time_ms:.1f}ms")
    print(f"[GEN3C] Mode: {result.metadata.get('mode', 'unknown')}")
    print(f"[GEN3C] Frames: {result.frame_count}")
    print(f"[GEN3C] Video path: {result.video_path}")

    if result.status == "success":
        print("[GEN3C] Test PASSED")
        return 0
    else:
        print(f"[GEN3C] Test FAILED: {result.error}")
        return 1


# ============================================================================
# CLI Parser
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gen3c_video.py",
        description="GEN3C: 3D-Consistent Video Generation",
    )
    parser.add_argument("--emit-bus", action="store_true", help="Emit events to pluribus bus")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # generate
    gen_p = sub.add_parser("generate", help="Generate video from views")
    gen_p.add_argument("--views", "-v", nargs="+", required=True, help="Input view images (supports glob)")
    gen_p.add_argument("--output", "-o", help="Output video path")
    gen_p.add_argument("--duration", "-d", type=float, default=4.0, help="Video duration in seconds")
    gen_p.add_argument("--fps", type=int, default=24, help="Frames per second")
    gen_p.add_argument("--width", type=int, default=512, help="Output width")
    gen_p.add_argument("--height", type=int, default=512, help="Output height")
    gen_p.add_argument("--camera", choices=["orbit", "linear", "spline"], default="orbit", help="Camera path type")
    gen_p.add_argument("--seed", type=int, default=-1, help="Random seed (-1 for random)")
    gen_p.set_defaults(func=cmd_generate)

    # serve
    serve_p = sub.add_parser("serve", help="Serve as HTTP endpoint")
    serve_p.add_argument("--port", type=int, default=9302, help="Port to serve on")
    serve_p.set_defaults(func=cmd_serve)

    # daemon
    daemon_p = sub.add_parser("daemon", help="Run as bus-driven daemon")
    daemon_p.set_defaults(func=cmd_daemon)

    # test
    test_p = sub.add_parser("test", help="Test video generation")
    test_p.add_argument("--views", nargs="*", help="Optional test views")
    test_p.set_defaults(func=cmd_test)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
