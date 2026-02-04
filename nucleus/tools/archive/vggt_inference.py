#!/usr/bin/env python3
"""VGGT Inference: Single-View 3D Scene Inference.

Wrapper for VGGT (Visual-Geometric-Grounding Transformer) for single-image 3D
reconstruction. Produces depth maps, surface normals, and triangle meshes.

Hardware Requirements:
- GPU Memory: 8GB+ (CUDA/ROCm)
- Inference Time: ~2s per image

Pluribus Integration:
- Dashboard: 3D visualization widgets (via WebGL/Three.js)
- Creative Tools: Spatial asset generation
- Bus Events: sota.vggt.request -> sota.vggt.response

Fallback Mode:
- When GPU is unavailable, generates mock depth/normal maps using gradient heuristics.

Usage:
    # Single image inference
    python3 vggt_inference.py infer --image /path/to/image.png

    # Serve as HTTP endpoint
    python3 vggt_inference.py serve --port 9301

    # Daemon mode (bus-driven)
    python3 vggt_inference.py daemon
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
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
        "actor": os.environ.get("PLURIBUS_ACTOR", "vggt-inference"),
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
class VGGTResult:
    """Result of VGGT inference."""
    image_path: str
    depth_map: bytes | None = None       # PNG bytes (grayscale depth)
    normal_map: bytes | None = None      # PNG bytes (RGB normals)
    mesh_obj: str | None = None          # OBJ format mesh string
    metadata: dict = field(default_factory=dict)
    inference_time_ms: float = 0.0
    status: str = "success"
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization (base64 encode binary data)."""
        return {
            "image_path": self.image_path,
            "depth_map_b64": base64.b64encode(self.depth_map).decode() if self.depth_map else None,
            "normal_map_b64": base64.b64encode(self.normal_map).decode() if self.normal_map else None,
            "mesh_obj": self.mesh_obj,
            "metadata": self.metadata,
            "inference_time_ms": self.inference_time_ms,
            "status": self.status,
            "error": self.error,
        }


# ============================================================================
# GPU Detection & Model Loading
# ============================================================================

_GPU_AVAILABLE: bool | None = None
_MODEL_LOADED: Any = None


def check_gpu_available() -> bool:
    """Check if CUDA/ROCm GPU is available."""
    global _GPU_AVAILABLE
    if _GPU_AVAILABLE is not None:
        return _GPU_AVAILABLE
    try:
        import torch
        _GPU_AVAILABLE = torch.cuda.is_available() or (hasattr(torch, "hip") and torch.hip.is_available())
    except ImportError:
        _GPU_AVAILABLE = False
    return _GPU_AVAILABLE


def load_vggt_model():
    """Load VGGT model (or return mock if unavailable)."""
    global _MODEL_LOADED
    if _MODEL_LOADED is not None:
        return _MODEL_LOADED

    if not check_gpu_available():
        print("[VGGT] GPU not available. Using fallback mock mode.", file=sys.stderr)
        _MODEL_LOADED = "mock"
        return _MODEL_LOADED

    try:
        # Attempt to load actual VGGT model
        # Note: VGGT may use different import paths depending on installation
        # This is a placeholder for the actual model loading
        import torch

        # Check for VGGT installation
        try:
            from vggt import VGGTModel  # type: ignore
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = VGGTModel.from_pretrained("facebook/vggt-large")
            model.to(device)
            model.eval()
            _MODEL_LOADED = model
            print(f"[VGGT] Model loaded on {device}.")
        except ImportError:
            # VGGT not installed, use fallback
            print("[VGGT] VGGT package not found. Using fallback mock mode.", file=sys.stderr)
            _MODEL_LOADED = "mock"

    except Exception as e:
        print(f"[VGGT] Model loading failed: {e}. Using fallback mock mode.", file=sys.stderr)
        _MODEL_LOADED = "mock"

    return _MODEL_LOADED


# ============================================================================
# Inference (Real & Mock)
# ============================================================================

def generate_mock_depth_map(width: int, height: int) -> bytes:
    """Generate a mock depth map using gradient heuristics."""
    try:
        from PIL import Image
        import numpy as np
        # Create a gradient depth map (center darker = closer)
        y, x = np.ogrid[:height, :width]
        center_x, center_y = width // 2, height // 2
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
        depth = (distance / max_dist * 255).astype(np.uint8)
        img = Image.fromarray(depth, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        # Fallback: return a tiny valid PNG (1x1 gray)
        return b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x00\x00\x00\x00:~\x9bU\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01\xe5\'\xde\xfc\x00\x00\x00\x00IEND\xaeB`\x82'


def generate_mock_normal_map(width: int, height: int) -> bytes:
    """Generate a mock normal map (flat surface facing camera)."""
    try:
        from PIL import Image
        import numpy as np
        # Normal map: flat surface facing camera is (0.5, 0.5, 1.0) in RGB
        normals = np.zeros((height, width, 3), dtype=np.uint8)
        normals[:, :, 0] = 128  # X = 0.5
        normals[:, :, 1] = 128  # Y = 0.5
        normals[:, :, 2] = 255  # Z = 1.0
        img = Image.fromarray(normals, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        return b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'


def generate_mock_mesh(width: int, height: int) -> str:
    """Generate a simple plane mesh in OBJ format."""
    # Create a simple quad mesh
    obj_lines = [
        "# VGGT Mock Mesh",
        f"# Generated: {now_iso_utc()}",
        "# Simple plane mesh (fallback mode)",
        "",
        "# Vertices",
        "v -1.0 -1.0 0.0",
        "v  1.0 -1.0 0.0",
        "v  1.0  1.0 0.0",
        "v -1.0  1.0 0.0",
        "",
        "# Texture coordinates",
        "vt 0.0 0.0",
        "vt 1.0 0.0",
        "vt 1.0 1.0",
        "vt 0.0 1.0",
        "",
        "# Normals",
        "vn 0.0 0.0 1.0",
        "",
        "# Faces",
        "f 1/1/1 2/2/1 3/3/1",
        "f 1/1/1 3/3/1 4/4/1",
    ]
    return "\n".join(obj_lines)


def infer_vggt(image_path: str | Path) -> VGGTResult:
    """Run VGGT inference on an image.

    Args:
        image_path: Path to input image (PNG/JPG)

    Returns:
        VGGTResult with depth map, normal map, and mesh
    """
    image_path = Path(image_path)
    if not image_path.exists():
        return VGGTResult(
            image_path=str(image_path),
            status="error",
            error=f"Image not found: {image_path}",
        )

    start_time = time.time()
    model = load_vggt_model()

    # Get image dimensions
    width, height = 512, 512
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception:
        pass

    if model == "mock":
        # Generate mock outputs
        depth_map = generate_mock_depth_map(width, height)
        normal_map = generate_mock_normal_map(width, height)
        mesh_obj = generate_mock_mesh(width, height)

        inference_time = (time.time() - start_time) * 1000

        return VGGTResult(
            image_path=str(image_path),
            depth_map=depth_map,
            normal_map=normal_map,
            mesh_obj=mesh_obj,
            metadata={
                "width": width,
                "height": height,
                "mode": "mock",
                "model": "vggt-fallback",
            },
            inference_time_ms=inference_time,
            status="success",
        )

    # Real model inference
    try:
        import torch
        from PIL import Image
        import numpy as np

        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        width, height = img.size

        # Model expects normalized tensor
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        device = next(model.parameters()).device
        img_tensor = img_tensor.to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(img_tensor)

        # Extract depth, normals, mesh from model outputs
        # Note: Actual output structure depends on VGGT implementation
        depth = outputs.get("depth", None)
        normals = outputs.get("normals", None)
        mesh = outputs.get("mesh", None)

        # Convert to output formats
        depth_map = None
        if depth is not None:
            depth_np = (depth[0].cpu().numpy() * 255).astype(np.uint8)
            depth_img = Image.fromarray(depth_np.squeeze(), mode="L")
            buf = io.BytesIO()
            depth_img.save(buf, format="PNG")
            depth_map = buf.getvalue()

        normal_map = None
        if normals is not None:
            normals_np = ((normals[0].cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
            normals_img = Image.fromarray(normals_np.transpose(1, 2, 0), mode="RGB")
            buf = io.BytesIO()
            normals_img.save(buf, format="PNG")
            normal_map = buf.getvalue()

        mesh_obj = None
        if mesh is not None:
            # Convert mesh to OBJ format
            vertices = mesh.get("vertices", [])
            faces = mesh.get("faces", [])
            lines = ["# VGGT Generated Mesh", f"# Generated: {now_iso_utc()}", ""]
            for v in vertices:
                lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
            lines.append("")
            for f in faces:
                lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}")
            mesh_obj = "\n".join(lines)

        inference_time = (time.time() - start_time) * 1000

        return VGGTResult(
            image_path=str(image_path),
            depth_map=depth_map,
            normal_map=normal_map,
            mesh_obj=mesh_obj,
            metadata={
                "width": width,
                "height": height,
                "mode": "gpu",
                "model": "vggt-large",
            },
            inference_time_ms=inference_time,
            status="success",
        )

    except Exception as e:
        return VGGTResult(
            image_path=str(image_path),
            status="error",
            error=str(e),
            inference_time_ms=(time.time() - start_time) * 1000,
        )


# ============================================================================
# CLI Commands
# ============================================================================

def cmd_infer(args: argparse.Namespace) -> int:
    """Run inference on a single image."""
    result = infer_vggt(args.image)

    if args.emit_bus:
        emit_bus_event(
            topic="sota.vggt.response",
            kind="inference",
            data=result.to_dict(),
            level="info" if result.status == "success" else "error",
        )

    if result.status != "success":
        print(f"[VGGT] Error: {result.error}", file=sys.stderr)
        return 1

    # Write outputs
    base = Path(args.image).stem
    out_dir = Path(args.output) if args.output else Path(args.image).parent

    if result.depth_map:
        depth_path = out_dir / f"{base}_depth.png"
        depth_path.write_bytes(result.depth_map)
        print(f"[VGGT] Depth map: {depth_path}")

    if result.normal_map:
        normal_path = out_dir / f"{base}_normals.png"
        normal_path.write_bytes(result.normal_map)
        print(f"[VGGT] Normal map: {normal_path}")

    if result.mesh_obj:
        mesh_path = out_dir / f"{base}.obj"
        mesh_path.write_text(result.mesh_obj, encoding="utf-8")
        print(f"[VGGT] Mesh: {mesh_path}")

    print(f"[VGGT] Inference complete in {result.inference_time_ms:.1f}ms (mode: {result.metadata.get('mode', 'unknown')})")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Serve VGGT inference as HTTP endpoint."""
    try:
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import urllib.parse
    except ImportError:
        print("[VGGT] HTTP server modules not available.", file=sys.stderr)
        return 1

    class VGGTHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok", "gpu": check_gpu_available()}).encode())
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path.startswith("/infer"):
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len)

                try:
                    data = json.loads(body)
                    image_path = data.get("image_path")
                    if not image_path:
                        raise ValueError("Missing image_path")

                    result = infer_vggt(image_path)

                    if args.emit_bus:
                        emit_bus_event(
                            topic="sota.vggt.response",
                            kind="inference",
                            data=result.to_dict(),
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
            print(f"[VGGT] {args[0]}")

    server = HTTPServer(("0.0.0.0", args.port), VGGTHandler)
    print(f"[VGGT] Serving on http://0.0.0.0:{args.port}")
    print(f"[VGGT] GPU available: {check_gpu_available()}")

    emit_bus_event(
        topic="sota.vggt.started",
        kind="service",
        data={"port": args.port, "gpu": check_gpu_available()},
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[VGGT] Shutting down...")
    return 0


def cmd_daemon(args: argparse.Namespace) -> int:
    """Run as bus-driven daemon."""
    print("[VGGT] Starting daemon mode...")
    print(f"[VGGT] GPU available: {check_gpu_available()}")
    print("[VGGT] Watching topic: sota.vggt.request")

    emit_bus_event(
        topic="sota.vggt.daemon.started",
        kind="service",
        data={"gpu": check_gpu_available()},
    )

    last_ts = time.time()

    while True:
        events = tail_bus_events("sota.vggt.request", since_ts=last_ts)

        for event in events:
            last_ts = max(last_ts, event.get("ts", 0))
            data = event.get("data", {})
            image_path = data.get("image_path")
            req_id = data.get("req_id", event.get("id", str(uuid.uuid4())))

            if not image_path:
                continue

            print(f"[VGGT] Processing request {req_id}: {image_path}")
            result = infer_vggt(image_path)

            emit_bus_event(
                topic="sota.vggt.response",
                kind="inference",
                data={
                    "req_id": req_id,
                    **result.to_dict(),
                },
                level="info" if result.status == "success" else "error",
            )

        time.sleep(0.5)


def cmd_test(args: argparse.Namespace) -> int:
    """Test inference with a sample image."""
    print("[VGGT] Running test inference...")
    print(f"[VGGT] GPU available: {check_gpu_available()}")

    # Create a test image if none provided
    test_image = Path(args.image) if args.image else None

    if test_image is None:
        try:
            from PIL import Image
            import numpy as np
            import tempfile

            # Create a simple test image (gradient)
            arr = np.zeros((256, 256, 3), dtype=np.uint8)
            arr[:, :, 0] = np.arange(256)  # Red gradient
            arr[:, :, 1] = np.arange(256).reshape(256, 1)  # Green gradient
            arr[:, :, 2] = 128  # Blue constant

            test_dir = Path(tempfile.gettempdir()) / "vggt_test"
            test_dir.mkdir(exist_ok=True)
            test_image = test_dir / "test_input.png"
            Image.fromarray(arr).save(test_image)
            print(f"[VGGT] Created test image: {test_image}")
        except ImportError:
            print("[VGGT] PIL not available. Please provide --image.", file=sys.stderr)
            return 1

    result = infer_vggt(test_image)

    print(f"[VGGT] Status: {result.status}")
    print(f"[VGGT] Inference time: {result.inference_time_ms:.1f}ms")
    print(f"[VGGT] Mode: {result.metadata.get('mode', 'unknown')}")
    print(f"[VGGT] Depth map: {len(result.depth_map)} bytes" if result.depth_map else "[VGGT] Depth map: None")
    print(f"[VGGT] Normal map: {len(result.normal_map)} bytes" if result.normal_map else "[VGGT] Normal map: None")
    print(f"[VGGT] Mesh: {len(result.mesh_obj)} chars" if result.mesh_obj else "[VGGT] Mesh: None")

    if result.status == "success":
        print("[VGGT] Test PASSED")
        return 0
    else:
        print(f"[VGGT] Test FAILED: {result.error}")
        return 1


# ============================================================================
# CLI Parser
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vggt_inference.py",
        description="VGGT: Single-View 3D Scene Inference",
    )
    parser.add_argument("--emit-bus", action="store_true", help="Emit events to pluribus bus")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # infer
    infer_p = sub.add_parser("infer", help="Run inference on an image")
    infer_p.add_argument("--image", "-i", required=True, help="Input image path")
    infer_p.add_argument("--output", "-o", help="Output directory")
    infer_p.set_defaults(func=cmd_infer)

    # serve
    serve_p = sub.add_parser("serve", help="Serve as HTTP endpoint")
    serve_p.add_argument("--port", type=int, default=9301, help="Port to serve on")
    serve_p.set_defaults(func=cmd_serve)

    # daemon
    daemon_p = sub.add_parser("daemon", help="Run as bus-driven daemon")
    daemon_p.set_defaults(func=cmd_daemon)

    # test
    test_p = sub.add_parser("test", help="Test inference")
    test_p.add_argument("--image", help="Optional test image")
    test_p.set_defaults(func=cmd_test)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
