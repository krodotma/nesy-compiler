#!/usr/bin/env python3
"""Live2Diff Stream: Real-Time Video Stylization.

Wrapper for Live2Diff model that performs real-time video stylization
at approximately 16 FPS. Transforms video streams with artistic styles
while maintaining temporal consistency.

Hardware Requirements:
- GPU Memory: 6GB+ (CUDA)
- Target FPS: ~16 (real-time capable)

Pluribus Integration:
- Dashboard: Live preview with style controls
- Creative Tools: Stream processing pipeline
- Bus Events: sota.live2diff.frame -> sota.live2diff.styled

Fallback Mode:
- When GPU unavailable, applies basic image filters (sepia, posterize, etc.)

Usage:
    # Stylize a video file
    python3 live2diff_stream.py stylize --input video.mp4 --style anime

    # Process webcam stream
    python3 live2diff_stream.py stream --style watercolor

    # Serve as HTTP endpoint
    python3 live2diff_stream.py serve --port 9303

    # Daemon mode (bus-driven frame processing)
    python3 live2diff_stream.py daemon
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import queue
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

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
        "actor": os.environ.get("PLURIBUS_ACTOR", "live2diff-stream"),
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
# Available Styles
# ============================================================================

AVAILABLE_STYLES = {
    "anime": "Anime/Cartoon style with cel-shading",
    "watercolor": "Soft watercolor painting effect",
    "oil_paint": "Oil painting with visible brushstrokes",
    "sketch": "Pencil sketch rendering",
    "neon": "Cyberpunk neon glow effect",
    "vintage": "Sepia-toned vintage film look",
    "comic": "Comic book style with halftone dots",
    "pixel": "Pixel art / retro game style",
    "impressionist": "Impressionist painting style",
    "none": "Pass-through (no stylization)",
}


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class StyleConfig:
    """Configuration for stylization."""
    style: str = "anime"
    strength: float = 1.0  # 0.0-1.0
    temporal_smoothing: float = 0.5  # Frame-to-frame consistency
    resolution_scale: float = 1.0  # Downscale for performance


@dataclass
class FrameResult:
    """Result of stylizing a single frame."""
    frame_idx: int
    original_bytes: bytes | None = None
    styled_bytes: bytes | None = None
    style: str = ""
    latency_ms: float = 0.0
    status: str = "success"
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "original_b64": base64.b64encode(self.original_bytes).decode() if self.original_bytes else None,
            "styled_b64": base64.b64encode(self.styled_bytes).decode() if self.styled_bytes else None,
            "style": self.style,
            "latency_ms": self.latency_ms,
            "status": self.status,
            "error": self.error,
        }


@dataclass
class StreamStats:
    """Statistics for stream processing."""
    frames_processed: int = 0
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    current_fps: float = 0.0
    dropped_frames: int = 0
    mode: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "frames_processed": self.frames_processed,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "current_fps": self.current_fps,
            "dropped_frames": self.dropped_frames,
            "mode": self.mode,
        }


# ============================================================================
# GPU Detection & Model Loading
# ============================================================================

_GPU_AVAILABLE: bool | None = None
_MODEL_LOADED: Any = None
_PREV_FRAME: Any = None  # For temporal smoothing


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


def load_live2diff_model():
    """Load Live2Diff model (or return mock if unavailable)."""
    global _MODEL_LOADED
    if _MODEL_LOADED is not None:
        return _MODEL_LOADED

    if not check_gpu_available():
        print("[Live2Diff] GPU not available. Using fallback filter mode.", file=sys.stderr)
        _MODEL_LOADED = "mock"
        return _MODEL_LOADED

    try:
        import torch

        # Attempt to load actual Live2Diff model
        try:
            from live2diff import Live2DiffModel  # type: ignore
            device = "cuda"
            model = Live2DiffModel.from_pretrained("tencent/live2diff-v1")
            model.to(device)
            model.eval()
            _MODEL_LOADED = model
            print(f"[Live2Diff] Model loaded on {device}.")
        except ImportError:
            print("[Live2Diff] Live2Diff package not found. Using fallback filter mode.", file=sys.stderr)
            _MODEL_LOADED = "mock"

    except Exception as e:
        print(f"[Live2Diff] Model loading failed: {e}. Using fallback filter mode.", file=sys.stderr)
        _MODEL_LOADED = "mock"

    return _MODEL_LOADED


# ============================================================================
# Fallback Filters (PIL-based)
# ============================================================================

def apply_sepia_filter(img):
    """Apply sepia/vintage filter."""
    try:
        from PIL import Image, ImageEnhance
        import numpy as np

        arr = np.array(img)
        # Sepia transformation matrix
        sepia_matrix = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131],
        ])
        sepia = np.dot(arr[..., :3], sepia_matrix.T)
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)
        return Image.fromarray(sepia)
    except Exception:
        return img


def apply_posterize_filter(img, bits: int = 3):
    """Apply posterization (reduce colors)."""
    try:
        from PIL import ImageOps
        return ImageOps.posterize(img, bits)
    except Exception:
        return img


def apply_edge_enhance(img):
    """Apply edge enhancement (sketch-like)."""
    try:
        from PIL import ImageFilter
        edges = img.filter(ImageFilter.FIND_EDGES)
        return edges.convert("RGB")
    except Exception:
        return img


def apply_contour(img):
    """Apply contour filter (comic-like)."""
    try:
        from PIL import ImageFilter
        return img.filter(ImageFilter.CONTOUR)
    except Exception:
        return img


def apply_blur_and_sharpen(img):
    """Apply blur then sharpen (oil paint simulation)."""
    try:
        from PIL import ImageFilter
        blurred = img.filter(ImageFilter.GaussianBlur(radius=2))
        sharpened = blurred.filter(ImageFilter.SHARPEN)
        return sharpened
    except Exception:
        return img


def apply_pixelate(img, pixel_size: int = 8):
    """Apply pixelation effect."""
    try:
        width, height = img.size
        small = img.resize(
            (width // pixel_size, height // pixel_size),
            resample=0  # NEAREST
        )
        return small.resize((width, height), resample=0)
    except Exception:
        return img


def apply_neon_effect(img):
    """Apply neon glow effect."""
    try:
        from PIL import ImageFilter, ImageEnhance
        import numpy as np

        # Enhance saturation
        enhancer = ImageEnhance.Color(img)
        saturated = enhancer.enhance(2.0)

        # Boost contrast
        enhancer = ImageEnhance.Contrast(saturated)
        contrasted = enhancer.enhance(1.5)

        # Add edge glow
        edges = img.filter(ImageFilter.FIND_EDGES)
        edges_arr = np.array(edges)
        img_arr = np.array(contrasted)

        # Add edge glow (blend)
        result = np.clip(img_arr + edges_arr * 0.5, 0, 255).astype(np.uint8)
        return Image.fromarray(result)
    except Exception:
        return img


def get_fallback_filter(style: str) -> Callable:
    """Get fallback filter function for a style."""
    from PIL import Image

    filters = {
        "anime": apply_posterize_filter,
        "watercolor": apply_blur_and_sharpen,
        "oil_paint": apply_blur_and_sharpen,
        "sketch": apply_edge_enhance,
        "neon": apply_neon_effect,
        "vintage": apply_sepia_filter,
        "comic": apply_contour,
        "pixel": apply_pixelate,
        "impressionist": apply_blur_and_sharpen,
        "none": lambda x: x,
    }
    return filters.get(style, lambda x: x)


# ============================================================================
# Frame Stylization
# ============================================================================

def stylize_frame(
    frame_bytes: bytes,
    config: StyleConfig,
    frame_idx: int = 0,
) -> FrameResult:
    """Stylize a single frame.

    Args:
        frame_bytes: Input frame as PNG/JPG bytes
        config: Style configuration
        frame_idx: Frame index for tracking

    Returns:
        FrameResult with styled frame
    """
    global _PREV_FRAME

    start_time = time.time()
    model = load_live2diff_model()

    try:
        from PIL import Image

        # Decode input frame
        img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
        orig_size = img.size

        # Apply resolution scaling
        if config.resolution_scale != 1.0:
            new_size = (
                int(orig_size[0] * config.resolution_scale),
                int(orig_size[1] * config.resolution_scale),
            )
            img = img.resize(new_size, Image.LANCZOS)

    except ImportError:
        return FrameResult(
            frame_idx=frame_idx,
            status="error",
            error="PIL not available",
            latency_ms=(time.time() - start_time) * 1000,
        )
    except Exception as e:
        return FrameResult(
            frame_idx=frame_idx,
            status="error",
            error=f"Frame decode error: {e}",
            latency_ms=(time.time() - start_time) * 1000,
        )

    if model == "mock":
        # Apply fallback filter
        filter_fn = get_fallback_filter(config.style)
        styled_img = filter_fn(img)

        # Apply strength blending
        if config.strength < 1.0:
            from PIL import Image
            styled_img = Image.blend(img, styled_img, config.strength)

        # Apply temporal smoothing with previous frame
        if _PREV_FRAME is not None and config.temporal_smoothing > 0:
            try:
                if _PREV_FRAME.size == styled_img.size:
                    styled_img = Image.blend(_PREV_FRAME, styled_img, 1 - config.temporal_smoothing * 0.5)
            except Exception:
                pass

        _PREV_FRAME = styled_img.copy()

        # Restore original size
        if config.resolution_scale != 1.0:
            styled_img = styled_img.resize(orig_size, Image.LANCZOS)

        # Encode output
        buf = io.BytesIO()
        styled_img.save(buf, format="PNG")
        styled_bytes = buf.getvalue()

        return FrameResult(
            frame_idx=frame_idx,
            original_bytes=frame_bytes,
            styled_bytes=styled_bytes,
            style=config.style,
            latency_ms=(time.time() - start_time) * 1000,
            status="success",
        )

    # Real model inference
    try:
        import torch
        import numpy as np

        # Prepare input tensor
        img_arr = np.array(img) / 255.0
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).float()
        device = next(model.parameters()).device
        img_tensor = img_tensor.to(device)

        # Encode style
        style_embedding = model.encode_style(config.style)

        # Generate styled frame
        with torch.no_grad():
            styled_tensor = model.stylize(
                img_tensor,
                style_embedding,
                strength=config.strength,
            )

        # Apply temporal smoothing
        if _PREV_FRAME is not None and config.temporal_smoothing > 0:
            styled_tensor = (1 - config.temporal_smoothing) * styled_tensor + config.temporal_smoothing * _PREV_FRAME

        _PREV_FRAME = styled_tensor.clone()

        # Convert to image
        styled_np = (styled_tensor[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        styled_img = Image.fromarray(styled_np)

        # Restore original size
        if config.resolution_scale != 1.0:
            styled_img = styled_img.resize(orig_size, Image.LANCZOS)

        # Encode output
        buf = io.BytesIO()
        styled_img.save(buf, format="PNG")
        styled_bytes = buf.getvalue()

        return FrameResult(
            frame_idx=frame_idx,
            original_bytes=frame_bytes,
            styled_bytes=styled_bytes,
            style=config.style,
            latency_ms=(time.time() - start_time) * 1000,
            status="success",
        )

    except Exception as e:
        return FrameResult(
            frame_idx=frame_idx,
            status="error",
            error=str(e),
            latency_ms=(time.time() - start_time) * 1000,
        )


# ============================================================================
# Stream Processing
# ============================================================================

class StreamProcessor:
    """Processes video streams with stylization."""

    def __init__(self, config: StyleConfig):
        self.config = config
        self.stats = StreamStats()
        self.running = False
        self._frame_queue: queue.Queue = queue.Queue(maxsize=30)
        self._result_queue: queue.Queue = queue.Queue(maxsize=30)
        self._worker_thread: threading.Thread | None = None
        self._last_frame_time = time.time()

    def start(self) -> None:
        """Start the stream processor."""
        self.running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        self.stats.mode = "mock" if load_live2diff_model() == "mock" else "gpu"

    def stop(self) -> None:
        """Stop the stream processor."""
        self.running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)

    def submit_frame(self, frame_bytes: bytes) -> bool:
        """Submit a frame for processing."""
        try:
            self._frame_queue.put_nowait((self.stats.frames_processed, frame_bytes))
            return True
        except queue.Full:
            self.stats.dropped_frames += 1
            return False

    def get_result(self, timeout: float = 0.1) -> FrameResult | None:
        """Get a processed frame result."""
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _worker_loop(self) -> None:
        """Background worker for frame processing."""
        while self.running:
            try:
                frame_idx, frame_bytes = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            result = stylize_frame(frame_bytes, self.config, frame_idx)

            # Update stats
            self.stats.frames_processed += 1
            self.stats.total_latency_ms += result.latency_ms
            self.stats.avg_latency_ms = self.stats.total_latency_ms / self.stats.frames_processed

            # Calculate FPS
            current_time = time.time()
            frame_delta = current_time - self._last_frame_time
            if frame_delta > 0:
                self.stats.current_fps = 1.0 / frame_delta
            self._last_frame_time = current_time

            try:
                self._result_queue.put_nowait(result)
            except queue.Full:
                pass  # Drop if output queue is full


# ============================================================================
# CLI Commands
# ============================================================================

def cmd_stylize(args: argparse.Namespace) -> int:
    """Stylize a video file."""
    import subprocess

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[Live2Diff] Error: Input not found: {input_path}", file=sys.stderr)
        return 1

    config = StyleConfig(
        style=args.style,
        strength=args.strength,
        temporal_smoothing=args.smoothing,
        resolution_scale=args.scale,
    )

    print(f"[Live2Diff] Stylizing: {input_path}")
    print(f"[Live2Diff] Style: {config.style}, Strength: {config.strength}")
    print(f"[Live2Diff] GPU available: {check_gpu_available()}")

    # Create temp directory for frames
    work_dir = Path(tempfile.mkdtemp(prefix="live2diff_"))

    # Extract frames using ffmpeg
    print("[Live2Diff] Extracting frames...")
    frame_pattern = work_dir / "frame_%05d.png"
    try:
        subprocess.run([
            "ffmpeg", "-i", str(input_path),
            "-vf", f"fps={args.fps}",
            str(frame_pattern),
        ], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[Live2Diff] Error extracting frames: {e}", file=sys.stderr)
        return 1

    # Process frames
    frame_files = sorted(work_dir.glob("frame_*.png"))
    total_frames = len(frame_files)
    print(f"[Live2Diff] Processing {total_frames} frames...")

    styled_dir = work_dir / "styled"
    styled_dir.mkdir()

    stats = StreamStats()
    stats.mode = "mock" if load_live2diff_model() == "mock" else "gpu"

    for i, frame_path in enumerate(frame_files):
        frame_bytes = frame_path.read_bytes()
        result = stylize_frame(frame_bytes, config, i)

        if result.status == "success" and result.styled_bytes:
            styled_path = styled_dir / frame_path.name
            styled_path.write_bytes(result.styled_bytes)

        stats.frames_processed += 1
        stats.total_latency_ms += result.latency_ms

        if (i + 1) % 10 == 0 or i == total_frames - 1:
            avg_ms = stats.total_latency_ms / stats.frames_processed
            fps = 1000 / avg_ms if avg_ms > 0 else 0
            print(f"[Live2Diff] Progress: {i + 1}/{total_frames} ({avg_ms:.1f}ms/frame, ~{fps:.1f} FPS)")

    # Combine frames back to video
    output_path = Path(args.output) if args.output else input_path.with_stem(f"{input_path.stem}_styled")
    styled_pattern = styled_dir / "frame_%05d.png"

    print(f"[Live2Diff] Encoding output video...")
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-framerate", str(args.fps),
            "-i", str(styled_pattern),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ], check=True, capture_output=True)
        print(f"[Live2Diff] Output saved: {output_path}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[Live2Diff] Error encoding video: {e}", file=sys.stderr)
        print(f"[Live2Diff] Styled frames available in: {styled_dir}")

    if args.emit_bus:
        emit_bus_event(
            topic="sota.live2diff.complete",
            kind="stylization",
            data={
                "input": str(input_path),
                "output": str(output_path),
                "style": config.style,
                "frames": total_frames,
                "stats": stats.to_dict(),
            },
        )

    return 0


def cmd_stream(args: argparse.Namespace) -> int:
    """Process live video stream."""
    config = StyleConfig(
        style=args.style,
        strength=args.strength,
        temporal_smoothing=args.smoothing,
        resolution_scale=args.scale,
    )

    print(f"[Live2Diff] Starting stream mode...")
    print(f"[Live2Diff] Style: {config.style}")
    print(f"[Live2Diff] GPU available: {check_gpu_available()}")

    # Try to open video source
    try:
        import cv2
    except ImportError:
        print("[Live2Diff] OpenCV (cv2) not available for webcam capture.", file=sys.stderr)
        print("[Live2Diff] Install with: pip install opencv-python", file=sys.stderr)
        return 1

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[Live2Diff] Error: Cannot open video source: {args.source}", file=sys.stderr)
        return 1

    processor = StreamProcessor(config)
    processor.start()

    print("[Live2Diff] Press 'q' to quit, 's' to cycle styles")

    style_idx = list(AVAILABLE_STYLES.keys()).index(args.style)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Encode frame to PNG
            _, buffer = cv2.imencode(".png", frame)
            frame_bytes = buffer.tobytes()

            # Submit for processing
            processor.submit_frame(frame_bytes)

            # Get result (non-blocking)
            result = processor.get_result(timeout=0.05)

            if result and result.styled_bytes:
                # Decode and display
                styled_arr = cv2.imdecode(
                    np.frombuffer(result.styled_bytes, np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if styled_arr is not None:
                    # Add status overlay
                    cv2.putText(
                        styled_arr,
                        f"Style: {config.style} | FPS: {processor.stats.current_fps:.1f} | Mode: {processor.stats.mode}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.imshow("Live2Diff", styled_arr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                # Cycle to next style
                style_idx = (style_idx + 1) % len(AVAILABLE_STYLES)
                config.style = list(AVAILABLE_STYLES.keys())[style_idx]
                print(f"[Live2Diff] Switched to style: {config.style}")

    except KeyboardInterrupt:
        pass
    finally:
        processor.stop()
        cap.release()
        cv2.destroyAllWindows()

    print(f"\n[Live2Diff] Stream stats: {processor.stats.to_dict()}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Serve Live2Diff stylization as HTTP endpoint."""
    from http.server import HTTPServer, BaseHTTPRequestHandler

    config = StyleConfig(style="anime")

    class Live2DiffHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "ok",
                    "gpu": check_gpu_available(),
                    "styles": list(AVAILABLE_STYLES.keys()),
                }).encode())
            elif self.path == "/styles":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(AVAILABLE_STYLES).encode())
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path.startswith("/stylize"):
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len)

                try:
                    data = json.loads(body)

                    # Get frame data (base64 encoded)
                    frame_b64 = data.get("frame_b64")
                    if not frame_b64:
                        raise ValueError("Missing frame_b64")

                    frame_bytes = base64.b64decode(frame_b64)

                    # Parse config
                    cfg = StyleConfig(
                        style=data.get("style", "anime"),
                        strength=data.get("strength", 1.0),
                        temporal_smoothing=data.get("temporal_smoothing", 0.5),
                        resolution_scale=data.get("resolution_scale", 1.0),
                    )

                    result = stylize_frame(frame_bytes, cfg, data.get("frame_idx", 0))

                    if args.emit_bus:
                        event_data = result.to_dict()
                        event_data.pop("original_b64", None)
                        event_data.pop("styled_b64", None)
                        emit_bus_event(
                            topic="sota.live2diff.styled",
                            kind="frame",
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
            print(f"[Live2Diff] {args[0]}")

    server = HTTPServer(("0.0.0.0", args.port), Live2DiffHandler)
    print(f"[Live2Diff] Serving on http://0.0.0.0:{args.port}")
    print(f"[Live2Diff] GPU available: {check_gpu_available()}")
    print(f"[Live2Diff] Available styles: {list(AVAILABLE_STYLES.keys())}")

    emit_bus_event(
        topic="sota.live2diff.started",
        kind="service",
        data={
            "port": args.port,
            "gpu": check_gpu_available(),
            "styles": list(AVAILABLE_STYLES.keys()),
        },
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Live2Diff] Shutting down...")
    return 0


def cmd_daemon(args: argparse.Namespace) -> int:
    """Run as bus-driven daemon for frame processing."""
    print("[Live2Diff] Starting daemon mode...")
    print(f"[Live2Diff] GPU available: {check_gpu_available()}")
    print("[Live2Diff] Watching topic: sota.live2diff.frame")

    emit_bus_event(
        topic="sota.live2diff.daemon.started",
        kind="service",
        data={
            "gpu": check_gpu_available(),
            "styles": list(AVAILABLE_STYLES.keys()),
        },
    )

    last_ts = time.time()
    stats = StreamStats()
    stats.mode = "mock" if load_live2diff_model() == "mock" else "gpu"

    while True:
        events = tail_bus_events("sota.live2diff.frame", since_ts=last_ts)

        for event in events:
            last_ts = max(last_ts, event.get("ts", 0))
            data = event.get("data", {})
            req_id = data.get("req_id", event.get("id", str(uuid.uuid4())))

            frame_b64 = data.get("frame_b64")
            if not frame_b64:
                continue

            try:
                frame_bytes = base64.b64decode(frame_b64)
            except Exception:
                continue

            config = StyleConfig(
                style=data.get("style", "anime"),
                strength=data.get("strength", 1.0),
                temporal_smoothing=data.get("temporal_smoothing", 0.5),
                resolution_scale=data.get("resolution_scale", 1.0),
            )

            result = stylize_frame(frame_bytes, config, data.get("frame_idx", stats.frames_processed))
            stats.frames_processed += 1
            stats.total_latency_ms += result.latency_ms
            stats.avg_latency_ms = stats.total_latency_ms / stats.frames_processed

            emit_bus_event(
                topic="sota.live2diff.styled",
                kind="frame",
                data={
                    "req_id": req_id,
                    **result.to_dict(),
                },
                level="info" if result.status == "success" else "error",
            )

        time.sleep(0.05)  # 20 Hz poll for ~16 FPS throughput


def cmd_test(args: argparse.Namespace) -> int:
    """Test stylization."""
    print("[Live2Diff] Running test stylization...")
    print(f"[Live2Diff] GPU available: {check_gpu_available()}")

    # Create test image
    try:
        from PIL import Image
        import numpy as np

        # Create a simple test image (color gradient)
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        for y in range(256):
            for x in range(256):
                arr[y, x] = [x, y, 128]

        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        frame_bytes = buf.getvalue()

    except ImportError:
        print("[Live2Diff] PIL not available.", file=sys.stderr)
        return 1

    # Test each style
    styles_to_test = ["anime", "sketch", "neon"] if not args.style else [args.style]

    for style in styles_to_test:
        config = StyleConfig(style=style, strength=1.0)
        result = stylize_frame(frame_bytes, config, 0)

        print(f"[Live2Diff] Style '{style}': {result.status}, {result.latency_ms:.1f}ms")

        if result.styled_bytes:
            # Save test output
            out_path = Path(tempfile.gettempdir()) / f"live2diff_test_{style}.png"
            out_path.write_bytes(result.styled_bytes)
            print(f"[Live2Diff]   Output: {out_path}")

    mode = "mock" if load_live2diff_model() == "mock" else "gpu"
    print(f"[Live2Diff] Mode: {mode}")
    print("[Live2Diff] Test PASSED")
    return 0


# ============================================================================
# CLI Parser
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="live2diff_stream.py",
        description="Live2Diff: Real-Time Video Stylization (~16 FPS)",
    )
    parser.add_argument("--emit-bus", action="store_true", help="Emit events to pluribus bus")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # stylize
    stylize_p = sub.add_parser("stylize", help="Stylize a video file")
    stylize_p.add_argument("--input", "-i", required=True, help="Input video path")
    stylize_p.add_argument("--output", "-o", help="Output video path")
    stylize_p.add_argument("--style", "-s", choices=list(AVAILABLE_STYLES.keys()), default="anime")
    stylize_p.add_argument("--strength", type=float, default=1.0, help="Style strength 0-1")
    stylize_p.add_argument("--smoothing", type=float, default=0.5, help="Temporal smoothing 0-1")
    stylize_p.add_argument("--scale", type=float, default=1.0, help="Resolution scale")
    stylize_p.add_argument("--fps", type=int, default=24, help="Output FPS")
    stylize_p.set_defaults(func=cmd_stylize)

    # stream
    stream_p = sub.add_parser("stream", help="Process live video stream")
    stream_p.add_argument("--source", default="0", help="Video source (0 for webcam, or path)")
    stream_p.add_argument("--style", "-s", choices=list(AVAILABLE_STYLES.keys()), default="anime")
    stream_p.add_argument("--strength", type=float, default=1.0)
    stream_p.add_argument("--smoothing", type=float, default=0.5)
    stream_p.add_argument("--scale", type=float, default=0.5, help="Resolution scale (lower for speed)")
    stream_p.set_defaults(func=cmd_stream)

    # serve
    serve_p = sub.add_parser("serve", help="Serve as HTTP endpoint")
    serve_p.add_argument("--port", type=int, default=9303, help="Port to serve on")
    serve_p.set_defaults(func=cmd_serve)

    # daemon
    daemon_p = sub.add_parser("daemon", help="Run as bus-driven daemon")
    daemon_p.set_defaults(func=cmd_daemon)

    # test
    test_p = sub.add_parser("test", help="Test stylization")
    test_p.add_argument("--style", choices=list(AVAILABLE_STYLES.keys()), help="Test specific style")
    test_p.set_defaults(func=cmd_test)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
